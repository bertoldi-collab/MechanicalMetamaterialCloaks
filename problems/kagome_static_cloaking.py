import dataclasses
import jax
import matplotlib.pyplot as plt
import jax.numpy as jnp
from jax import jacobian, jit, random, value_and_grad, vmap, flatten_util, grad
import nlopt
from dataclasses import dataclass
from typing import Any, Dict, List, Literal, Optional, Union
from mechanicalmetamaterialcloaks.dynamics import setup_dynamic_solver
from mechanicalmetamaterialcloaks.energy import (
    build_contact_energy,
    build_strain_energy,
    combine_block_energies,
    ligament_energy,
    ligament_energy_linearized,
)
from mechanicalmetamaterialcloaks.geometry import (
    CloakKagomeGeometry,
    KagomeGeometry,
    compute_edge_angles,
    compute_edge_lengths,
)

from mechanicalmetamaterialcloaks.utils import (
    ContactParams,
    ControlParams,
    GeometricalParams,
    LigamentParams,
    MechanicalParams,
    SolutionData,
    SolutionType,
)


@dataclass
class ForwardInput:
    """Input params for the forward solve function."""

    # Geometry
    shifts_1: Any
    shifts_2: Any
    shifts_3: Any

    # Dynamic loading
    # amplitude: Any
    # loading_rate: Any


@dataclass
class ForwardProblem:
    """
    Forward problem for the static cloaking of a void in a Kagome metamaterial.
    BCs:
        - Left column clamped.
        - Right column slowly driven in compression.
    """

    # KagomeGeometry
    n1_cells: int
    n2_cells: int
    bond_length: Any
    spacing: float

    # Cloak Geometry
    void: List
    width_strip_cloak_area: float

    # Mechanical
    k_stretch: Any
    k_shear: Any
    k_rot: Any
    density: Any
    damping: Any

    # Contact
    k_contact: Any
    min_angle: Any
    cutoff_angle: Any

    # Dynamic loading
    amplitude: Any

    # Analysis params
    simulation_time: Any
    n_timepoints: int
    linearized_strains: bool = False

    # Solver params
    atol: float = 1e-8
    rtol: float = 1e-8

    # Mother Geometry
    shifts_1_2_3_mg: Any = None  # one of the two argument needs to be fill

    # Problem name
    name: str = "kagome_static_cloaking"

    # Solution or list of solutions
    solution_data: Optional[Union[SolutionType, List[SolutionType]]] = None

    # Flag indicating that solve method is not available. It needs to be set up by calling self.setup().
    is_setup: bool = False

    # Other cloak geometry params
    middle_cloak_area: Optional[jnp.ndarray] = None

    def setup(self) -> None:
        """
        Set up forward solver.
        """

        # Set up solver of the intact geometry
        print("Setup solver of the intact geometry")
        # Displacement field of the mother geometry
        self.mother_geometry = KagomeGeometry(
            n1_cells=self.n1_cells,
            n2_cells=self.n2_cells,
            bond_length=self.bond_length,
            direct_basis=self.spacing * jnp.eye(2),
        )

        self.mother_geometry.compute_geometry()

        # centroid_node_vectors is a function of the initial angle
        (
            self.block_centroids_mg,
            self.centroid_node_vectors_mg,
            self.bond_connectivity_mg,
            self.reference_bond_vectors_mg,
        ) = self.mother_geometry.get_parametrization()

        _init_centroid_node_vectors_mg = self.centroid_node_vectors_mg(
            *self.shifts_1_2_3_mg
        )
        _init_block_centroids_mg = self.block_centroids_mg(*self.shifts_1_2_3_mg)

        _bond_connectivity_mg = self.bond_connectivity_mg()
        _reference_bond_vectors_mg = self.reference_bond_vectors_mg()

        # cloaked geometry
        self.cloaked_geometry = CloakKagomeGeometry(
            self.mother_geometry,
            initial_block_centroids_mother_geometry=_init_block_centroids_mg,
            void=self.void,
            width_strip_cloak_area=self.width_strip_cloak_area,
            middle_cloak_area=self.middle_cloak_area,
        )

        # Initial conditions
        state0_mg = jnp.array(
            [
                0
                * random.uniform(
                    random.PRNGKey(0), (self.mother_geometry.n_blocks, 3)
                ),  # Initial position
                0
                * random.uniform(
                    random.PRNGKey(1), (self.mother_geometry.n_blocks, 3)
                ),  # Initial velocity
            ]
        )

        # Damping
        damped_blocks_mg = jnp.arange(self.mother_geometry.n_blocks)
        self.damping_mg = self.damping * jnp.ones((self.mother_geometry.n_blocks, 3))

        # Dynamic input and BCs
        mgIDs_blocks_last_column = (
            jnp.arange(0, 2 * self.mother_geometry.n2_cells)
            // 2
            * 2
            * self.mother_geometry.n1_cells
            + 2 * self.mother_geometry.n1_cells
            - 1
            - jnp.arange(0, 2 * self.mother_geometry.n2_cells) % 2
        )
        driven_block_DOF_pairs_mg = jnp.array(
            [
                jnp.tile(mgIDs_blocks_last_column, 3),
                jnp.array(
                    [0] * self.n2_cells * 2 + [1] * self.n2_cells * 2 + [2] * self.n2_cells * 2
                ),
            ]
        ).T
        mgIDs_blocks_first_column = (
            jnp.arange(0, 2 * self.n2_cells) // 2 * 2 * self.mother_geometry.n1_cells
            + jnp.arange(0, 2 * self.mother_geometry.n2_cells) % 2
        )
        clamped_blocks_mg = jnp.array(
            [
                jnp.tile(mgIDs_blocks_first_column, 3),
                jnp.array(
                    [0] * self.n2_cells * 2 + [1] * self.n2_cells * 2 + [2] * self.n2_cells * 2
                ),
            ]
        ).T

        constrained_block_DOF_pairs_mg = jnp.concatenate(
            [driven_block_DOF_pairs_mg, clamped_blocks_mg]
        ).astype(int)

        # Dynamic loading
        def applied_displacement(t, amplitude, simulation_time):
            return amplitude * (1 + jnp.tanh(6 * (2 * t / simulation_time - 1))) / 2

        self.applied_displacement = applied_displacement

        def constrained_DOFs_fn_mg(t, amplitude, simulation_time):
            fn = jnp.zeros((len(constrained_block_DOF_pairs_mg),))
            fn = fn.at[: self.n2_cells * 2].set(
                -1.0 * applied_displacement(t, amplitude, simulation_time)
            )
            return fn

        # Construct strain energy
        strain_energy_mg = build_strain_energy(
            bond_connectivity=_bond_connectivity_mg,
            bond_energy_fn=ligament_energy_linearized
            if self.linearized_strains
            else ligament_energy,
        )
        contact_energy_mg = build_contact_energy(
            bond_connectivity=_bond_connectivity_mg
        )
        potential_energy_mg = combine_block_energies(
            strain_energy_mg, contact_energy_mg
        )

        # elastic force useful to compute the reaction force
        self.elastic_forces_mg = grad(potential_energy_mg)

        # Control Parameters for Dynamic Solver
        control_params_mg = ControlParams(
            geometrical_params=GeometricalParams(
                block_centroids=_init_block_centroids_mg,
                centroid_node_vectors=_init_centroid_node_vectors_mg,
            ),
            mechanical_params=MechanicalParams(
                bond_params=LigamentParams(
                    k_stretch=self.k_stretch,
                    k_shear=self.k_shear,
                    k_rot=self.k_rot,
                    reference_vector=_reference_bond_vectors_mg,
                ),
                density=self.density,
                damping=self.damping_mg,
                contact_params=ContactParams(
                    min_angle=self.min_angle,
                    cutoff_angle=self.cutoff_angle,
                    k_contact=self.k_contact,
                ),
            ),
            constraint_params=dict(
                amplitude=self.amplitude, simulation_time=self.simulation_time
            ),
        )
        self.control_params_mg = control_params_mg

        # Setup solver
        solve_dynamics_MG = setup_dynamic_solver(
            geometry=self.mother_geometry,
            energy_fn=potential_energy_mg,
            constrained_block_DOF_pairs=constrained_block_DOF_pairs_mg,
            constrained_DOFs_fn=constrained_DOFs_fn_mg,
            damped_blocks=damped_blocks_mg,
            atol=self.atol,
            rtol=self.rtol,
        )

        # Analysis params
        simulation_time = self.simulation_time
        n_timepoints = self.n_timepoints
        timepoints = jnp.linspace(0, simulation_time, n_timepoints)

        # Solve dynamics
        print("Compute dynamics of the intact geometry")
        solution_mg = (
            solve_dynamics_MG(
                state0=state0_mg,
                timepoints=timepoints,
                control_params=control_params_mg,
            )
            if self.solution_data is None
            else self.solution_data[0].fields
        )

        # Save data
        solutionData_mg = SolutionData(
            block_centroids=_init_block_centroids_mg,
            centroid_node_vectors=_init_centroid_node_vectors_mg,
            bond_connectivity=_bond_connectivity_mg,
            timepoints=timepoints,
            fields=solution_mg,
        )
        self.solutionData_mg_fields = solutionData_mg.fields
        self.solutionData_mg = solutionData_mg

        # Build the solver dynamics of the cloak geometry
        print("Setup solver of the voided geometry")
        # Build cloaked geometry
        self.cloaked_geometry.compute_geometry()
        (
            self.block_centroids_cg,
            self.centroid_node_vectors_cg,
            self.bond_connectivity_cg,
            self.reference_bond_vectors_cg,
        ) = self.cloaked_geometry.get_parametrization()
        self._bond_connectivity_cg = self.bond_connectivity_cg()
        self._reference_bond_vectors_cg = self.reference_bond_vectors_cg()

        # Damping
        damped_blocks_cg = jnp.arange(self.cloaked_geometry.n_blocks)
        self.damping_cg = self.damping * jnp.ones((self.cloaked_geometry.n_blocks, 3))

        # Dynamic input and BCs
        ids_driven_block_DOF_pairs_cg_first_column = jnp.array(
            [
                self.cloaked_geometry.grid_convert_mgIDs_to_cgIDs[k, 0]
                for k in range(0, self.n2_cells)
            ]
            + [
                self.cloaked_geometry.grid_convert_mgIDs_to_cgIDs[k, 1]
                for k in range(0, self.n2_cells)
            ]
        )
        ids_clamped_block_DOF_pairs_cg_last_column = jnp.array(
            [
                self.cloaked_geometry.grid_convert_mgIDs_to_cgIDs[
                    k, 2 * self.n1_cells - 1
                ]
                for k in range(self.n2_cells)
            ]
            + [
                self.cloaked_geometry.grid_convert_mgIDs_to_cgIDs[
                    k, 2 * self.n1_cells - 2
                ]
                for k in range(self.n2_cells)
            ]
        )

        driven_block_DOF_pairs_cg = jnp.array(
            [
                jnp.tile(ids_clamped_block_DOF_pairs_cg_last_column, 3),
                jnp.array(
                    [0] * self.n2_cells * 2
                    + [1] * self.n2_cells * 2
                    + [2] * self.n2_cells * 2
                ),
            ]
        ).T
        clamped_blocks_cg = jnp.array(
            [
                jnp.tile(ids_driven_block_DOF_pairs_cg_first_column, 3),
                jnp.array(
                    [0] * self.n2_cells * 2
                    + [1] * self.n2_cells * 2
                    + [2] * self.n2_cells * 2
                ),
            ]
        ).T
        self.driven_block_DOF_pairs_cg = driven_block_DOF_pairs_cg

        constrained_block_DOF_pairs_cg = jnp.concatenate(
            [driven_block_DOF_pairs_cg, clamped_blocks_cg]
        ).astype(int)

        def constrained_DOFs_fn_cg(t, amplitude, simulation_time):
            fn = jnp.zeros((len(constrained_block_DOF_pairs_cg),))
            fn = fn.at[: self.n2_cells * 2].set(
                -1.0 * applied_displacement(t, amplitude, simulation_time)
            )
            return fn

        # Construct strain energy
        strain_energy_cg = build_strain_energy(
            bond_connectivity=self._bond_connectivity_cg,
            bond_energy_fn=ligament_energy_linearized
            if self.linearized_strains
            else ligament_energy,
        )
        contact_energy_cg = build_contact_energy(
            bond_connectivity=self._bond_connectivity_cg
        )
        potential_energy_cg = combine_block_energies(
            strain_energy_cg, contact_energy_cg
        )
        # elastic force useful to compute the reaction force
        self.elastic_forces_cg = grad(potential_energy_cg)

        # Setup solver
        solve_dynamics_cg = setup_dynamic_solver(
            geometry=self.cloaked_geometry,
            energy_fn=potential_energy_cg,
            constrained_block_DOF_pairs=constrained_block_DOF_pairs_cg,
            constrained_DOFs_fn=constrained_DOFs_fn_cg,
            damped_blocks=damped_blocks_cg,
            atol=self.atol,
            rtol=self.rtol,
        )

        def plot_sketch():
            # Plots to visualize region of the cloak and void
            blocks_for_plot = self.block_centroids_cg(*self.shifts_1_2_3_mg)
            fig, axes = plt.subplots(constrained_layout=True)
            axes.plot(*jnp.array(self.void).T)
            axes.scatter(
                blocks_for_plot[self.cloaked_geometry.cgIDs_cloak_area, 0],
                blocks_for_plot[self.cloaked_geometry.cgIDs_cloak_area, 1],
                c="red",
            )
            axes.scatter(
                blocks_for_plot[self.cloaked_geometry.cgIDs_surronding_area, 0],
                blocks_for_plot[self.cloaked_geometry.cgIDs_surronding_area, 1],
                c="green",
            )
            axes.axis("equal")
            return fig, axes

        # Utility functions to map between all (mother) and cloak shifts
        def all_to_cloak_shifts(all_shifts):
            shifts_1, shifts_2, shifts_3 = all_shifts
            return (
                shifts_1[self.cloaked_geometry.mask_shifts_1],
                shifts_2[self.cloaked_geometry.mask_shifts_2],
                shifts_3[self.cloaked_geometry.mask_shifts_3],
            )

        def cloak_to_all_shifts(cloak_shifts):
            return self.cloaked_geometry.get_shifts_whole_area_in_2DLatticeGeometry(
                cloak_shifts, self.shifts_1_2_3_mg
            )

        # Setup forward
        # Initial conditions
        state0 = jnp.array(
            [
                jnp.zeros((self.cloaked_geometry.n_blocks, 3)),  # Initial position
                jnp.zeros((self.cloaked_geometry.n_blocks, 3)),  # Initial velocity
            ]
        )

        def get_control_params_cg_all_shifts(shifts_1_2_3):
            shifts_1, shifts_2, shifts_3 = shifts_1_2_3
            # Change the control params given the new geometry
            _block_centroids_cg = self.block_centroids_cg(shifts_1, shifts_2, shifts_3)
            _centroid_node_vectors_cg = self.centroid_node_vectors_cg(
                shifts_1, shifts_2, shifts_3
            )

            control_params_cg = ControlParams(
                geometrical_params=GeometricalParams(
                    block_centroids=_block_centroids_cg,
                    centroid_node_vectors=_centroid_node_vectors_cg,
                ),
                mechanical_params=MechanicalParams(
                    bond_params=LigamentParams(
                        k_stretch=self.k_stretch,
                        k_shear=self.k_shear,
                        k_rot=self.k_rot,
                        reference_vector=self._reference_bond_vectors_cg,
                    ),
                    density=self.density,
                    damping=self.damping_cg,
                    contact_params=ContactParams(
                        min_angle=self.min_angle,
                        cutoff_angle=self.cutoff_angle,
                        k_contact=self.k_contact,
                    ),
                ),
                constraint_params=dict(
                    amplitude=self.amplitude, simulation_time=self.simulation_time
                ),
            )
            return control_params_cg

        # Control params fn
        def get_control_params_cg(shifts_cloak_area):
            shifts_1, shifts_2, shifts_3 = cloak_to_all_shifts(shifts_cloak_area)
            return get_control_params_cg_all_shifts((shifts_1, shifts_2, shifts_3))

        def forward_all_shifts(shifts_1_2_3):
            control_params_cg = get_control_params_cg_all_shifts(shifts_1_2_3)

            # Solve dynamics
            solution = solve_dynamics_cg(
                state0=state0, timepoints=timepoints, control_params=control_params_cg
            )

            return SolutionData(
                block_centroids=control_params_cg.geometrical_params.block_centroids,
                centroid_node_vectors=control_params_cg.geometrical_params.centroid_node_vectors,
                bond_connectivity=self._bond_connectivity_cg,
                timepoints=timepoints,
                fields=solution,
            )

        # Forward function
        def forward(shifts_cloak_area):
            shifts_1, shifts_2, shifts_3 = cloak_to_all_shifts(shifts_cloak_area)
            return forward_all_shifts((shifts_1, shifts_2, shifts_3))

        self.solve = forward
        self.solve_all_shifts = forward_all_shifts
        self.is_setup = True
        self.all_to_cloak_shifts = all_to_cloak_shifts
        self.cloak_to_all_shifts = cloak_to_all_shifts
        self.get_control_params_cg = get_control_params_cg
        self.get_control_params_cg_all_shifts = get_control_params_cg_all_shifts
        self.plot_sketch = plot_sketch
        print("Initialization done")

    def _force_displacement(
        self, solution_data: SolutionData, control_params: ControlParams
    ):
        if self.is_setup:
            displacement_history = solution_data.fields[:, 0]
            block_DOF_pairs = self.driven_block_DOF_pairs_cg[: self.n2_cells]
            force_history = vmap(
                lambda u: jnp.sum(
                    self.elastic_forces_cg(u, control_params)[
                        block_DOF_pairs[:, 0], block_DOF_pairs[:, 1]
                    ]
                )
            )(displacement_history)
            applied_u = self.applied_displacement(
                solution_data.timepoints,
                self.amplitude,
                self.simulation_time,
            )
            return jnp.array([applied_u, force_history])

    def force_displacement_cloak(self, solution_data: SolutionData, shifts_cloak):
        return self._force_displacement(
            solution_data, self.get_control_params_cg(shifts_cloak)
        )

    @staticmethod
    def from_data(problem_data):
        problem_data = ForwardProblem(**problem_data)
        problem_data.is_setup = False
        return problem_data

    def to_data(self):
        return ForwardProblem(**dataclasses.asdict(self))

    @staticmethod
    def from_dict(dict_in):
        # Convert solution data to named tuple
        if dict_in["solution_data"] is not None:
            if type(dict_in["solution_data"]) is dict:
                dict_in["solution_data"] = SolutionData(
                    **dict_in["solution_data"])
            elif type(dict_in["solution_data"]) is list:
                dict_in["solution_data"] = [SolutionData(
                    **solution) for solution in dict_in["solution_data"]]
        problem_data = ForwardProblem(**dict_in)
        problem_data.is_setup = False
        return problem_data

    def to_dict(self):
        # Make sure namedtuples are converted to dictionaries before saving
        dict_out = dataclasses.asdict(self)
        if type(dict_out["solution_data"]) is SolutionData:
            dict_out["solution_data"] = dict_out["solution_data"]._asdict()
        elif type(dict_out["solution_data"]) is list:
            dict_out["solution_data"] = [solution._asdict()
                                         for solution in dict_out["solution_data"]]
        return dict_out


@dataclass
class OptimizationProblem:
    """
    docstring
    """

    # Forward problem provides a forward function that can be called on design variables
    forward_problem: ForwardProblem
    forward_input: ForwardInput
    objective_type: Literal["integrated", "final"] = "final"
    objective_values: Optional[List[Any]] = None
    design_values: Optional[List[Any]] = None
    constraints_violation: Optional[Dict[str, List[Any]]] = None
    name: str = ForwardProblem.name

    # Flag indicating that objective_fn method is not available. It needs to be set up by calling self.setup_objective().
    is_setup: bool = False

    def __post_init__(self):
        self.objective_values = (
            [] if self.objective_values is None else self.objective_values
        )
        self.design_values = [] if self.design_values is None else self.design_values
        self.constraints_violation = (
            {"angles": [], "edge_lengths": []}
            if self.constraints_violation is None
            else self.constraints_violation
        )

    def setup_objective(self) -> None:
        """
        Jit compiles the objective function.
        """

        # Make sure forward solvers are set up
        if not self.forward_problem.is_setup:
            self.forward_problem.setup()

        self.dimension_less = jnp.array(
            [
                1 / self.forward_problem.spacing,
                1 / self.forward_problem.spacing,
                1,
            ]
        )

        if self.objective_type == "integrated":
            self.max_delta_mg = (
                (
                    self.forward_problem.solutionData_mg_fields[
                        -1,
                        0,
                        self.forward_problem.cloaked_geometry.mgIDs_surronding_area,
                        :,
                    ]
                    * self.dimension_less
                )
                ** 2
            ).sum() ** 0.5

            # Delta function integrated over time to match configuration along the entire time interval
            def delta_function_integrated(shifts_cloak_area):
                """
                Args :
                    shifts_cloak_area: tuple[jnp.ndarray, jnp.ndarray]
                        More precisely, if horizontal_shift, vertical_shift = shifts_cloak_area
                        then     horizontal_shift(jnp.ndarray of shape(geometry_cloak.nb_horizontal_shifts,2))
                            and  vertical_shift((jnp.ndarray of shape(geometry_cloak.nb_vertical_shifts,2)))
                """
                # Solve forward
                solution_data_cg = self.forward_problem.solve(shifts_cloak_area)

                return (
                    (
                        (
                            (
                                solution_data_cg.fields[
                                    :,
                                    0,
                                    self.forward_problem.cloaked_geometry.cgIDs_surronding_area,
                                    :,
                                ]
                                - self.forward_problem.solutionData_mg_fields[
                                    :,
                                    0,
                                    self.forward_problem.cloaked_geometry.mgIDs_surronding_area,
                                    :,
                                ]
                            )
                            * self.dimension_less
                        )
                        ** 2
                    ).sum()
                ) ** 0.5 / self.max_delta_mg
        elif self.objective_type == "final":
            self.max_delta_mg = jnp.max(
                (
                    (
                        self.forward_problem.solutionData_mg_fields[
                            :,
                            0,
                            self.forward_problem.cloaked_geometry.mgIDs_surronding_area,
                            :,
                        ]
                        * self.dimension_less
                    )
                    ** 2
                ).sum(axis=(1, 2))
                ** 0.5
            )

            # Delta function at the final time to match final configuration
            def delta_function_final(shifts_cloak_area):
                """
                Args :
                    shifts_cloak_area: tuple[jnp.ndarray, jnp.ndarray]
                        More precisely, if horizontal_shift, vertical_shift = shifts_cloak_area
                        then     horizontal_shift(jnp.ndarray of shape(geometry_cloak.nb_horizontal_shifts,2))
                            and  vertical_shift((jnp.ndarray of shape(geometry_cloak.nb_vertical_shifts,2)))
                """
                # Solve forward
                solution_data_cg = self.forward_problem.solve(shifts_cloak_area)

                return (
                    (
                        (
                            (
                                solution_data_cg.fields[
                                    -1,
                                    0,
                                    self.forward_problem.cloaked_geometry.cgIDs_surronding_area,
                                    :,
                                ]
                                - self.forward_problem.solutionData_mg_fields[
                                    -1,
                                    0,
                                    self.forward_problem.cloaked_geometry.mgIDs_surronding_area,
                                    :,
                                ]
                            )
                            * self.dimension_less
                        )
                        ** 2
                    ).sum()
                ) ** 0.5 / self.max_delta_mg
        else:
            raise ValueError(
                f"Objective type {self.objective_type} not recognized. Must be 'integrated' or 'final'."
            )

        self.objective_fn = (
            delta_function_final
            if self.objective_type == "final"
            else delta_function_integrated
        )
        self.is_setup = True

    # TODO: change accordingly
    def setup_angle_constraints(
        self, min_void_angle=0.0, min_block_angle=0.0, max_block_angle=None
    ):
        # centroid_node_vectors = self.forward_problem.cloaked_geometry.centroid_node_vectors

        if max_block_angle is None:

            def angle_constraints(shifts_1_2_3):
                _centroid_node_vectors = (
                    self.forward_problem.cloaked_geometry.centroid_node_vectors(
                        *shifts_1_2_3
                    )
                )
                _bond_connectivity_cg = self.forward_problem._bond_connectivity_cg

                void_angles_1, void_angles_2, block_angles_1, block_angles_2 = vmap(
                    lambda b: jnp.mod(
                        jnp.array(compute_edge_angles(_centroid_node_vectors, b)),
                        2 * jnp.pi,
                    ),
                    in_axes=0,
                )(_bond_connectivity_cg).T

                return jnp.concatenate(
                    [
                        -(void_angles_1 - min_void_angle),
                        -(void_angles_2 - min_void_angle),
                        -(block_angles_1 - min_block_angle),
                        -(block_angles_2 - min_block_angle),
                    ]
                )
        else:

            def angle_constraints(shifts_1_2_3):
                _centroid_node_vectors = (
                    self.forward_problem.cloaked_geometry.centroid_node_vectors(
                        *shifts_1_2_3
                    )
                )
                _bond_connectivity_cg = self.forward_problem._bond_connectivity_cg

                void_angles_1, void_angles_2, block_angles_1, block_angles_2 = vmap(
                    lambda b: jnp.mod(
                        jnp.array(compute_edge_angles(_centroid_node_vectors, b)),
                        2 * jnp.pi,
                    ),
                    in_axes=0,
                )(_bond_connectivity_cg).T

                return jnp.concatenate(
                    [
                        -(void_angles_1 - min_void_angle),
                        -(void_angles_2 - min_void_angle),
                        -(block_angles_1 - min_block_angle),
                        -(block_angles_2 - min_block_angle),
                        (block_angles_1 - max_block_angle),
                        (block_angles_2 - max_block_angle),
                    ]
                )

        self.angle_constraints = lambda cloak_shifts: angle_constraints(
            self.forward_problem.cloak_to_all_shifts(cloak_shifts)
        )

    def setup_edge_length_constraints(self, min_edge_length):
        def edge_length_constraints(shifts_1_2_3):
            edge_lengths = compute_edge_lengths(
                self.forward_problem.cloaked_geometry.centroid_node_vectors(
                    *shifts_1_2_3
                )
            ).reshape(-1)
            return -(edge_lengths - min_edge_length)

        self.edge_length_constraints = lambda cloak_shifts: edge_length_constraints(
            self.forward_problem.cloak_to_all_shifts(cloak_shifts)
        )

    def run_optimization_nlopt(
        self,
        initial_guess,
        n_iterations: int,
        max_time: Optional[int] = None,
        lower_bound: Optional[float] = None,
        upper_bound: Optional[float] = None,
        min_void_angle: Optional[float] = None,
        min_block_angle: Optional[float] = None,
        max_block_angle: Optional[float] = None,
        min_edge_length: Optional[float] = None,
    ):
        # Make sure objective_fn is set up
        if not self.is_setup:
            self.setup_objective()

        def flatten(tree):
            return flatten_util.ravel_pytree(tree)[0]

        _, unflatten = flatten_util.ravel_pytree(initial_guess)

        objective_and_grad = jit(value_and_grad(self.objective_fn))
        # objective_and_grad = value_and_grad(self.objective_fn)

        def nlopt_objective(x, grad):
            v, g = objective_and_grad(unflatten(x))  # jax evaluation

            self.objective_values.append(v)
            self.design_values.append(unflatten(x))

            print(
                f"Iteration: {len(self.objective_values)}\nObjective = {self.objective_values[-1]}"
            )

            if grad.size > 0:
                grad[:] = flatten(g)

            return float(v)

        initial_guess_flattened = flatten(initial_guess)

        opt = nlopt.opt(nlopt.LD_MMA, len(initial_guess_flattened))

        if min_void_angle is not None and min_block_angle is not None:
            self.setup_angle_constraints(
                min_void_angle, min_block_angle, max_block_angle
            )
            angle_constraints = jit(lambda x: self.angle_constraints(unflatten(x)))
            angle_constraints_jac = jit(
                jacobian(lambda x: self.angle_constraints(unflatten(x)))
            )

            def nlopt_angle_constraints(result, x, grad):
                result[:] = angle_constraints(x)

                self.constraints_violation["angles"].append(result.max())

                print(
                    f"Angle constraints violation = {self.constraints_violation['angles'][-1]}"
                )

                if grad.size > 0:
                    grad[:, :] = angle_constraints_jac(x)

            opt.add_inequality_mconstraint(
                nlopt_angle_constraints,
                1.0e-8
                * jnp.ones(
                    (4 * len(self.forward_problem._bond_connectivity_cg),)
                    if max_block_angle is None
                    else (6 * len(self.forward_problem._bond_connectivity_cg),)
                ),
            )

        if min_edge_length is not None:
            self.setup_edge_length_constraints(min_edge_length)
            edge_length_constraints = jit(
                lambda x: self.edge_length_constraints(unflatten(x))
            )
            edge_length_constraints_jac = jit(
                jacobian(lambda x: self.edge_length_constraints(unflatten(x)))
            )

            def nlopt_edge_length_constraints(result, x, grad):
                result[:] = edge_length_constraints(x)
                self.constraints_violation["edge_lengths"].append(result.max())

                print(
                    f"Edge length constraints violation = {self.constraints_violation['edge_lengths'][-1]}"
                )

                if grad.size > 0:
                    grad[:, :] = edge_length_constraints_jac(x)

            opt.add_inequality_mconstraint(
                nlopt_edge_length_constraints,
                1.0e-8
                * jnp.ones(
                    self.forward_problem.cloaked_geometry.n_blocks
                    * self.forward_problem.cloaked_geometry.n_npb
                ),
            )

        opt.set_param("verbosity", 1)
        opt.set_maxeval(n_iterations)

        opt.set_min_objective(nlopt_objective)

        if lower_bound is not None:
            opt.set_lower_bounds(lower_bound)
        if upper_bound is not None:
            opt.set_upper_bounds(upper_bound)

        if max_time is not None:
            opt.set_maxtime(max_time)

        opt.optimize(initial_guess_flattened)

        # Store forward solution data for the intact geometry, the initial design, and the optimized design
        self.compute_best_forward()

    def compute_best_forward(self):
        """Compute the forward solution for the intact geometry, the initial design, and the optimized design."""

        if len(self.design_values) == 0:
            raise ValueError("No design has been optimized yet.")

        if not self.forward_problem.is_setup:
            self.forward_problem.setup()

        self.forward_problem.solution_data = []
        self.forward_problem.solution_data.append(
            self.forward_problem.solutionData_mg
        )  # Add the mother geometry solution
        self.forward_problem.solution_data.append(
            self.forward_problem.solve(self.design_values[0])
        )  # Add the initial design solution
        self.forward_problem.solution_data.append(
            self.forward_problem.solve(self.design_values[-1])
        )  # Add the optimized design solution

        return self.forward_problem.solution_data

    def reaction_force_history(
        self,
        column_type: str = "first",
        displacement_history: jnp.array = None,
        geometry_type: str = "mg",
        type_force: str = "resulting",
    ):
        """
        Returns the reaction force history of the first or last column of the mother geometry or the cloaked geometry.

        Args:
            - column_type(str): should be "first" or "last"
            - displacement_history(jnp.array): could be the solutionData.fields[:,0,:,:] value.
                    Two execptions:
                        1 - for the mother geometry, displacement_history will be retrieved from the attribute of the forward_problem object.
                        2 - if None when it is computed for initial guess or optimized cloaked geometry, then the compute_best_forward method
                            is used to compute the displacement history.
            - geometry_type(str): should be "mg", "ig" or "opcg".
            - type_force(str): should be either "resulting" or "local"

        """

        if not self.forward_problem.is_setup:
            self.forward_problem.setup()

        if type_force == "resulting":
            def aggregation_fn(u): return jnp.sum(u)
        elif type_force == "local":
            def aggregation_fn(u): return u

        # reaction force for the mother geometry
        if geometry_type == "mg":
            displacement_history = self.forward_problem.solutionData_mg.fields[
                :, 0, :, :
            ]
            if column_type == "first":
                ids_DOF_pairs_mg = (
                    jnp.arange(0, self.forward_problem.n2_cells)
                    * self.forward_problem.n1_cells
                )

            elif column_type == "last":
                ids_DOF_pairs_mg = (
                    jnp.arange(0, self.forward_problem.n2_cells)
                    * self.forward_problem.n1_cells
                    + self.forward_problem.n1_cells
                    - 1
                )

            block_DOF_pairs_mg_targeted_column = jnp.array(
                [
                    jnp.tile(ids_DOF_pairs_mg, 1),
                    jnp.array([0] * self.forward_problem.n2_cells),
                ]
            ).T

            fs = vmap(
                lambda u: aggregation_fn(
                    self.forward_problem.elastic_forces_mg(
                        u, self.forward_problem.control_params_mg
                    )[
                        block_DOF_pairs_mg_targeted_column[:, 0],
                        block_DOF_pairs_mg_targeted_column[:, 1],
                    ]
                )
            )(displacement_history)
            us = self.forward_problem.applied_displacement(
                jnp.linspace(
                    0,
                    self.forward_problem.simulation_time,
                    self.forward_problem.n_timepoints,
                ),
                self.forward_problem.amplitude,
                self.forward_problem.simulation_time,
            )
            return us, fs

        # reqction force for the cloacked geometry
        elif geometry_type in ["opcg", "ig"]:
            # if displacement_history is None, then the compute_best_forward method is used to compute the displacement history
            if displacement_history == None:
                solution_data = (
                    self.compute_best_forward()
                    if self.forward_problem.solution_data is None
                    else self.forward_problem.solution_data
                )
                if geometry_type == "opcg":
                    displacement_history = solution_data[2].fields[:, 0, :, :]
                elif geometry_type == "ig":
                    displacement_history = solution_data[1].fields[:, 0, :, :]

            if geometry_type == "opcg":
                _control_params_cg = self.forward_problem.get_control_params_cg(
                    self.design_values[0]
                )
            elif geometry_type == "ig":
                _control_params_cg = self.forward_problem.get_control_params_cg(
                    self.design_values[-1]
                )

            n1_cells = self.forward_problem.mother_geometry.n1_cells
            n2_cells = self.forward_problem.mother_geometry.n2_cells

            if column_type == "first":
                ids_DOF_pairs_cg = jnp.array(
                    [
                        self.forward_problem.cloaked_geometry.grid_convert_mgIDs_to_cgIDs[
                            k, 0
                        ]
                        for k in range(n2_cells)
                    ]
                )
            elif column_type == "last":
                ids_DOF_pairs_cg = jnp.array(
                    [
                        self.forward_problem.cloaked_geometry.grid_convert_mgIDs_to_cgIDs[
                            k, n1_cells - 1
                        ]
                        for k in range(n2_cells)
                    ]
                )

            nb_blocks_in_column = ids_DOF_pairs_cg.shape[0]
            block_DOF_pairs_cg_targeted_column = jnp.array(
                [jnp.tile(ids_DOF_pairs_cg, 1), jnp.array([0] * nb_blocks_in_column)]
            ).T

            fs = vmap(
                lambda u: aggregation_fn(
                    self.forward_problem.elastic_forces_cg(u, _control_params_cg)[
                        block_DOF_pairs_cg_targeted_column[:, 0],
                        block_DOF_pairs_cg_targeted_column[:, 1],
                    ]
                )
            )(displacement_history)
            us = self.forward_problem.applied_displacement(
                jnp.linspace(
                    0,
                    self.forward_problem.simulation_time,
                    self.forward_problem.n_timepoints,
                ),
                self.forward_problem.amplitude,
                self.forward_problem.simulation_time,
            )
            return us, fs

        # double check if values are correct
        if column_type not in ["first", "last"]:
            raise ValueError(
                f"column_type {column_type} not recognized. Must be 'first' or 'last'."
            )
        if geometry_type not in ["mg", "ig", "opcg"]:
            raise ValueError(
                f"geometry_type {geometry_type} not recognized. Must be 'mg' or 'ig' or 'opcg'."
            )
        if type_force not in ["resulting", "local"]:
            raise ValueError(
                f"type_force {type_force} not recognized. Must be 'resulting' to get the resulting force on the column, or 'local' to get the local force."
            )

    @staticmethod
    def from_data(optimization_data):
        optimization_data.forward_problem = ForwardProblem.from_data(
            optimization_data.forward_problem
        )
        optimization_data.forward_input = ForwardInput(
            **optimization_data.forward_input
        )
        optimization_data.is_setup = False
        return optimization_data

    def to_data(self):
        return OptimizationProblem(**dataclasses.asdict(self))

    @staticmethod
    def from_dict(dict_in):
        # Convert solution data to named tuple
        dict_in["forward_problem"] = ForwardProblem.from_dict(dict_in["forward_problem"])
        dict_in["forward_input"] = ForwardInput(**dict_in["forward_input"])
        # Ensure design values are iterable of jax arrays
        dict_in["design_values"] = jax.tree.map(
            lambda x: jnp.array(x), dict_in["design_values"]
        )
        optimization_data = OptimizationProblem(**dict_in)
        optimization_data.is_setup = False
        return optimization_data

    def to_dict(self):
        # Make sure namedtuples are converted to dictionaries before saving
        dict_out = dataclasses.asdict(self)
        dict_out["forward_problem"] = self.forward_problem.to_dict()
        return dict_out
