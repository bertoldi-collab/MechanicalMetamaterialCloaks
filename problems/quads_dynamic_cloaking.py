import dataclasses
import jax
import matplotlib.pyplot as plt
import jax.numpy as jnp
from jax import jacobian, jit, random, value_and_grad, vmap, flatten_util
import nlopt
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Union
from mechanicalmetamaterialcloaks.dynamics import setup_dynamic_solver
from mechanicalmetamaterialcloaks.energy import (
    build_contact_energy,
    build_strain_energy,
    combine_block_energies,
    ligament_energy,
    ligament_energy_linearized,
)
from mechanicalmetamaterialcloaks.geometry import (
    CloakQuadGeometry,
    QuadGeometry,
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
    horizontal_shifts: Any  # initial guess horizontal shifts
    vertical_shifts: Any  # initial guess vertical shifts

    # Dynamic loading
    # amplitude: Any
    # loading_rate: Any


@dataclass
class ForwardProblem:
    """
    Forward problem for the dynamic cloaking of a void in a quad metamaterial.
    BCs:
        - Left column driven harmonically.
    """

    # QuadGeometry
    n1_blocks: int
    n2_blocks: int
    spacing: float
    bond_length: float

    # Cloak Geometry
    void: List
    width_strip_cloak_area: float

    # Mechanical
    k_stretch: float
    k_shear: float
    k_rot: float
    density: float
    damping_xy_factor: float
    damping_rot_factor: float

    # Contact
    k_contact: float
    min_angle: float
    cutoff_angle: Any

    # Dynamic loading
    amplitude: float
    loading_rate: float
    tau: float
    input_delay: float
    n_pulses: int

    # Analysis params
    simulation_time: float
    n_timepoints: int
    linearized_strains: bool = False

    # Mother Geometry
    # at least one of the two argument needs to be different to None
    # by default horizontal_vertical_shifts_mg is taken into account in priority
    horizontal_vertical_shifts_mg: Any = None
    initial_angle: Optional[float] = None

    # Problem name
    name: str = "quads_dynamic_cloaking"

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
        # First, the dynamics of the mother geometry is computed.
        # mother geometry object contains attributes and methods to characterized the parametrization of nodes and bounds
        mother_geometry = QuadGeometry(
            n1_blocks=self.n1_blocks,
            n2_blocks=self.n2_blocks,
            spacing=self.spacing,
            bond_length=self.bond_length,
        )
        self.mother_geometry = mother_geometry

        # centroid_node_vectors is a function of the initial angle
        (
            block_centroids_mg,
            centroid_node_vectors_mg,
            bond_connectivity_mg,
            reference_bond_vectors_mg,
        ) = mother_geometry.get_parametrization()

        # initialize the position of centroids and nodes
        # below two methods are accepted
        if self.horizontal_vertical_shifts_mg != None:
            # 1. by providing specific nodes shift
            _init_centroid_node_vectors_mg = centroid_node_vectors_mg(
                *self.horizontal_vertical_shifts_mg
            )
            _init_block_centroids_mg = block_centroids_mg(
                *self.horizontal_vertical_shifts_mg
            )
        elif self.initial_angle != None:
            # 2. or initialize a grid of rotated squares
            self.horizontal_vertical_shifts_mg = (
                mother_geometry.get_design_from_rotated_square(angle=self.initial_angle)
            )
            _init_centroid_node_vectors_mg = centroid_node_vectors_mg(
                *self.horizontal_vertical_shifts_mg
            )
            _init_block_centroids_mg = block_centroids_mg(
                *self.horizontal_vertical_shifts_mg
            )
        else:
            raise ValueError(
                "At least matrix of nodes' shifts or an initial angle should be "
            )

        # initialize connections between quads
        _bond_connectivity_mg = bond_connectivity_mg()
        _reference_bond_vectors_mg = reference_bond_vectors_mg()

        # Initial conditions
        state0_mg = jnp.array(
            [
                0
                * random.uniform(
                    random.PRNGKey(0), (mother_geometry.n_blocks, 3)
                ),  # Initial position
                0
                * random.uniform(
                    random.PRNGKey(1), (mother_geometry.n_blocks, 3)
                ),  # Initial velocity
            ]
        )

        # Damping
        damped_blocks_mg = jnp.arange(mother_geometry.n_blocks)

        self.damping_mg = (
            0.0186
            * jnp.array(
                [
                    self.damping_xy_factor
                    * 2
                    * (0.36125 * self.density * self.spacing**2 * self.k_shear) ** 0.5,
                    self.damping_xy_factor
                    * 2
                    * (0.36125 * self.density * self.spacing**2 * self.k_shear) ** 0.5,
                    self.damping_rot_factor
                    * 2
                    * (0.02175026 * self.density * self.spacing**4 * self.k_rot) ** 0.5,
                ]
            )
            * jnp.ones((self.n1_blocks * self.n2_blocks, 3))
        )

        # Dynamic input and BCs
        # all blocks of the first column are driven
        constrained_block_DOF_pairs_mg = jnp.array(
            [
                jnp.tile(jnp.arange(0, self.n2_blocks) * self.n1_blocks, 3),
                jnp.array(
                    [0] * self.n2_blocks + [1] * self.n2_blocks + [2] * self.n2_blocks
                ),
            ]
        ).T

        # making sure int type is respected
        constrained_block_DOF_pairs_mg = constrained_block_DOF_pairs_mg.astype(int)
        driven_blocks_ids_mg = jnp.unique(constrained_block_DOF_pairs_mg[:, 0])
        moving_blocks_ids_mg = jnp.setdiff1d(
            jnp.arange(mother_geometry.n_blocks), driven_blocks_ids_mg
        )

        # constrain all degrees of freedom of the driven blocks
        constrained_DOFs_loading_vector_mg = jnp.zeros(
            (len(constrained_block_DOF_pairs_mg),)
        )

        # allow displacement over the horizontal axis
        constrained_DOFs_loading_vector_mg = constrained_DOFs_loading_vector_mg.at[
            : self.n2_blocks
        ].set(1)

        # define the type of loading
        def harmonic_ramp(t, amplitude, loading_rate, tau):
            return (
                amplitude
                * (1 + jnp.tanh(t / tau))
                / 2
                * jnp.sin(2 * jnp.pi * loading_rate * t)
            )

        # apply the loading on the driven blocks
        def constrained_DOFs_fn_mg(t, amplitude, loading_rate, input_delay, tau):
            return (
                harmonic_ramp(t - input_delay, amplitude, loading_rate, tau)
                * constrained_DOFs_loading_vector_mg
            )

        # Construct energy of the system
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

        # Control Parameters for Dynamic Solver
        # control_params_mg encapsulates all useful params to solve dynamcis of mother geometry
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
                amplitude=self.amplitude,
                loading_rate=self.loading_rate,
                input_delay=self.input_delay,
                tau=self.tau,
            ),
        )

        # Setup solver
        solve_dynamics_MG = setup_dynamic_solver(
            geometry=mother_geometry,
            energy_fn=potential_energy_mg,
            constrained_block_DOF_pairs=constrained_block_DOF_pairs_mg,
            constrained_DOFs_fn=constrained_DOFs_fn_mg,
            damped_blocks=damped_blocks_mg,
            atol=1e-4,
        )
        print("Setup of the solver for mother geometry done")

        # Analysis params
        simulation_time = self.simulation_time
        n_timepoints = self.n_timepoints
        timepoints = jnp.linspace(0, simulation_time, n_timepoints)

        # Solve dynamics of the mother geometry
        solution_mg = solve_dynamics_MG(
            state0=state0_mg, timepoints=timepoints, control_params=control_params_mg
        )

        print("Solution of the dynamics of the mother geometry done")

        # Save data
        solutionData_mg = SolutionData(
            block_centroids=_init_block_centroids_mg,
            centroid_node_vectors=_init_centroid_node_vectors_mg,
            bond_connectivity=_bond_connectivity_mg,
            timepoints=timepoints,
            fields=solution_mg,
        )
        self.solutionData_mg = solutionData_mg

        # Build the solver dynamics of the cloak geometry
        # Define cloak area
        cloaked_geometry = CloakQuadGeometry(
            mother_geometry,
            initial_block_centroids_mother_geometry=_init_block_centroids_mg,
            void=self.void,
            width_strip_cloak_area=self.width_strip_cloak_area,
        )
        # to_cloak=True,
        # add_extra_damped_columns=True)
        self.cloaked_geometry = cloaked_geometry

        # initialize the position of centroids and nodes
        (
            block_centroids_cg,
            centroid_node_vectors_cg,
            bond_connectivity_cg,
            reference_bond_vectors_cg,
        ) = cloaked_geometry.get_parametrization()

        self._bond_connectivity_cg = bond_connectivity_cg()
        self._reference_bond_vectors_cg = reference_bond_vectors_cg()

        # block_centroids_cg and centroid_node_vectors_cg are functions of the nodes' shift
        self.block_centroids_cg = block_centroids_cg
        self.centroid_node_vectors_cg = centroid_node_vectors_cg

        # damping
        damped_blocks_cg = jnp.arange(self.cloaked_geometry.n_blocks)
        self.damping_cg = (
            0.0186
            * jnp.array(
                [
                    self.damping_xy_factor
                    * 2
                    * (0.36125 * self.density * self.spacing**2 * self.k_shear) ** 0.5,
                    self.damping_xy_factor
                    * 2
                    * (0.36125 * self.density * self.spacing**2 * self.k_shear) ** 0.5,
                    self.damping_rot_factor
                    * 2
                    * (0.02175026 * self.density * self.spacing**4 * self.k_rot) ** 0.5,
                ]
            )
            * jnp.ones((self.cloaked_geometry.n_blocks, 3))
        )

        # dynamic input and BCs
        # all blocks of the first column are driven
        ids_driven_block_DOF_pairs_cg = jnp.array(
            [
                self.cloaked_geometry.grid_convert_mgIDs_to_cgIDs[k, 0]
                for k in range(self.cloaked_geometry.n2_blocks)
            ]
        )

        constrained_block_DOF_pairs_cg = jnp.array(
            [
                jnp.tile(ids_driven_block_DOF_pairs_cg, 3),
                jnp.array(
                    [0] * self.n2_blocks + [1] * self.n2_blocks + [2] * self.n2_blocks
                ),
            ]
        ).T

        # making sure that int type is respected
        constrained_block_DOF_pairs_cg = constrained_block_DOF_pairs_cg.astype(int)
        driven_blocks_ids_cg = jnp.unique(constrained_block_DOF_pairs_cg[:, 0])
        moving_blocks_ids_cg = jnp.setdiff1d(
            jnp.arange(cloaked_geometry.n_blocks), driven_blocks_ids_cg
        )

        # constrain all degrees of freedom of the driven blocks
        constrained_DOFs_loading_vector_cg = jnp.zeros(
            (len(constrained_block_DOF_pairs_cg),)
        )

        # allow displacement over the horizontal axis
        constrained_DOFs_loading_vector_cg = constrained_DOFs_loading_vector_cg.at[
            : self.n2_blocks
        ].set(1)

        # apply the loading on the driven blocks
        def constrained_DOFs_fn_cg(t, amplitude, loading_rate, input_delay, tau):
            return (
                harmonic_ramp(t - input_delay, amplitude, loading_rate, tau)
                * constrained_DOFs_loading_vector_cg
            )

        # Construct energy of the system
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

        # Setup solver
        solve_dynamics_cg = setup_dynamic_solver(
            geometry=self.cloaked_geometry,
            energy_fn=potential_energy_cg,
            constrained_block_DOF_pairs=constrained_block_DOF_pairs_cg,
            constrained_DOFs_fn=constrained_DOFs_fn_cg,
            damped_blocks=damped_blocks_cg,
            atol=1e-4,
        )
        print("Setup of the solver for cloak geometry done")

        def plot_sketch():
            # Plots to visualize region of the cloak and void
            blocks_for_plot = block_centroids_cg(*self.horizontal_vertical_shifts_mg)
            fig, axes = plt.subplots(constrained_layout=True)
            axes.plot(*jnp.array(self.void).T)
            axes.scatter(
                blocks_for_plot[
                    self.cloaked_geometry.cgIDs_cloak_area, 0
                ],
                blocks_for_plot[
                    self.cloaked_geometry.cgIDs_cloak_area, 1
                ],
                c="red",
            )
            axes.scatter(
                blocks_for_plot[
                    self.cloaked_geometry.cgIDs_surronding_area, 0
                ],
                blocks_for_plot[
                    self.cloaked_geometry.cgIDs_surronding_area, 1
                ],
                c="green",
            )
            axes.axis("equal")
            return fig, axes

        # Utility functions to map between all (mother) and cloak shifts
        def all_to_cloak_shifts(all_shifts):
            horizontal_shifts, vertical_shifts = all_shifts
            return horizontal_shifts[
                self.cloaked_geometry.mask_horizontal_shift_allowed
            ], vertical_shifts[self.cloaked_geometry.mask_vertical_shift_allowed]

        def cloak_to_all_shifts(cloak_shifts):
            return self.cloaked_geometry.get_horizontal_vertical_shifts_whole_area_in_2DLatticeGeometry(
                cloak_shifts, self.horizontal_vertical_shifts_mg
            )

        # Setup forward at frequency and amplitude for mother geometry
        def forward_mg_at_amplitude_frequency(
            amplitude,
            frequency,
        ):
            # Initial conditions
            state0 = jnp.zeros((2, self.mother_geometry.n_blocks, 3))

            # Control Parameters for Dynamic Solver

            control_params_mg = ControlParams(
                geometrical_params=GeometricalParams(
                    block_centroids=block_centroids_mg(
                        *self.horizontal_vertical_shifts_mg
                    ),
                    centroid_node_vectors=centroid_node_vectors_mg(
                        *self.horizontal_vertical_shifts_mg
                    ),
                ),
                mechanical_params=MechanicalParams(
                    bond_params=LigamentParams(
                        k_stretch=self.k_stretch,
                        k_shear=self.k_shear,
                        k_rot=self.k_rot,
                        reference_vector=reference_bond_vectors_mg(),
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
                    amplitude=amplitude,
                    loading_rate=frequency,
                    input_delay=self.input_delay,
                    tau=self.tau,
                ),
            )

            # Solve dynamics
            solution = solve_dynamics_MG(
                state0=state0, timepoints=timepoints, control_params=control_params_mg
            )

            return SolutionData(
                block_centroids=block_centroids_mg(
                    *self.horizontal_vertical_shifts_mg
                ),
                centroid_node_vectors=centroid_node_vectors_mg(
                    *self.horizontal_vertical_shifts_mg
                ),
                bond_connectivity=_bond_connectivity_mg,
                timepoints=timepoints,
                fields=solution,
            )

        # Setup forward for cloak geometry
        def forward(horizontal_vertical_shifts_cloak_area):
            horizontal_shifts, vertical_shifts = cloak_to_all_shifts(
                horizontal_vertical_shifts_cloak_area
            )

            # Initial conditions
            state0 = jnp.zeros((2, self.cloaked_geometry.n_blocks, 3))

            # Change the control params given the new geometry
            _block_centroids_cg = self.block_centroids_cg(
                horizontal_shifts, vertical_shifts
            )
            _centroid_node_vectors_cg = self.centroid_node_vectors_cg(
                horizontal_shifts, vertical_shifts
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
                    amplitude=self.amplitude,
                    loading_rate=self.loading_rate,
                    input_delay=self.input_delay,
                    tau=self.tau,
                ),
            )

            # Solve dynamics
            solution = solve_dynamics_cg(
                state0=state0, timepoints=timepoints, control_params=control_params_cg
            )

            return SolutionData(
                block_centroids=_block_centroids_cg,
                centroid_node_vectors=_centroid_node_vectors_cg,
                bond_connectivity=self._bond_connectivity_cg,
                timepoints=timepoints,
                fields=solution,
            )

        # Setup forward at frequency and amplitude for cloak geometry
        def forward_cg_at_amplitude_frequency(
            horizontal_vertical_shifts_cloak_area,
            amplitude,
            frequency,
        ):
            horizontal_shifts, vertical_shifts = cloak_to_all_shifts(
                horizontal_vertical_shifts_cloak_area
            )

            # Initial conditions
            state0 = jnp.zeros((2, self.cloaked_geometry.n_blocks, 3))

            # Change the control params given the new geometry
            _block_centroids_cg = self.block_centroids_cg(
                horizontal_shifts, vertical_shifts
            )
            _centroid_node_vectors_cg = self.centroid_node_vectors_cg(
                horizontal_shifts, vertical_shifts
            )

            # Control Parameters for Dynamic Solver

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
                    amplitude=amplitude,
                    loading_rate=frequency,
                    input_delay=self.input_delay,
                    tau=self.tau,
                ),
            )

            # Solve dynamics
            solution = solve_dynamics_cg(
                state0=state0, timepoints=timepoints, control_params=control_params_cg
            )

            return SolutionData(
                block_centroids=_block_centroids_cg,
                centroid_node_vectors=_centroid_node_vectors_cg,
                bond_connectivity=self._bond_connectivity_cg,
                timepoints=timepoints,
                fields=solution,
            )

        self.solve = forward
        self.solve_cg_at_amplitude_frequency = forward_cg_at_amplitude_frequency
        self.solve_mg_at_amplitude_frequency = forward_mg_at_amplitude_frequency
        self.is_setup = True
        self.all_to_cloak_shifts = all_to_cloak_shifts
        self.cloak_to_all_shifts = cloak_to_all_shifts
        self.plot_sketch = plot_sketch
        self.moving_blocks_ids_mg = moving_blocks_ids_mg
        self.driven_blocks_ids_mg = driven_blocks_ids_mg
        self.moving_blocks_ids_cg = moving_blocks_ids_cg
        self.driven_blocks_ids_cg = driven_blocks_ids_cg

        print("Setup ForwardProblem done")

    @staticmethod
    def from_dict(dict_in):
        # Convert solution data to named tuple
        if dict_in["solution_data"] is not None:
            if type(dict_in["solution_data"]) is dict:
                dict_in["solution_data"] = {
                    key: SolutionData(
                        **solution) if type(solution) is not list else [SolutionData(**s) for s in solution]
                    for key, solution in dict_in["solution_data"].items()
                }
            elif type(dict_in["solution_data"]) is list:
                dict_in["solution_data"] = [
                    SolutionData(**solution) for solution in dict_in["solution_data"]
                ]
        dict_in["horizontal_vertical_shifts_mg"] = jax.tree.map(
            lambda x: jnp.array(x),
            dict_in["horizontal_vertical_shifts_mg"],
        )
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
        elif type(dict_out["solution_data"]) is dict:
            dict_out["solution_data"] = {
                key: solution._asdict() if type(solution) is not list else [s._asdict() for s in solution]
                for key, solution in dict_out["solution_data"].items()
            }
        return dict_out


@dataclass
class OptimizationProblem:
    """
    docstring
    """

    # Forward problem provides a forward function that can be called on design variables
    forward_problem: ForwardProblem
    forward_input: ForwardInput
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

        # making sure forward solver is setup
        if not self.forward_problem.is_setup:
            self.forward_problem.setup()

        # useful constants
        self.dimension_less = jnp.array(
            [
                1 / self.forward_problem.cloaked_geometry.spacing,
                1 / self.forward_problem.cloaked_geometry.spacing,
                1,
            ]
        )
        self.max_delta_mg = jnp.max(
            (
                (
                    self.forward_problem.solutionData_mg.fields[
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

        # define objective function
        def delta_function_integrated(horizontal_vertical_shifts_cloak_area):
            """
            Args :
                horizontal_vertical_shifts_cloak_area: tuple[jnp.ndarray, jnp.ndarray]
                    Precisely, if horizontal_shift, vertical_shift = horizontal_vertical_shifts_cloak_area
                    then horizontal_shift is a jnp.ndarray object of shape: (geometry_cloak.nb_horizontal_shifts,2))
                    and vertical_shift is a jnp.ndarray object of shape: (geometry_cloak.nb_vertical_shifts,2)))
            """
            # Solve forward problem
            solution_data_cg = self.forward_problem.solve(
                horizontal_vertical_shifts_cloak_area
            )

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
                            - self.forward_problem.solutionData_mg.fields[
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

        def delta_function_integrated_from_fields(fields_mg, fields_cg):
            return (
                (
                    (
                        (
                            fields_cg[
                                :,
                                0,
                                self.forward_problem.cloaked_geometry.cgIDs_surronding_area,
                                :,
                            ]
                            - fields_mg[
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

        self.objective_fn = delta_function_integrated
        self.objective_fn_from_fields = delta_function_integrated_from_fields
        self.is_setup = True

    def setup_angle_constraints(self, min_void_angle=0.0, min_block_angle=0.0):
        def angle_constraints(horizontal_vertical_shifts):
            _centroid_node_vectors = (
                self.forward_problem.cloaked_geometry.centroid_node_vectors(
                    *horizontal_vertical_shifts
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

        self.angle_constraints = lambda cloak_shifts: angle_constraints(
            self.forward_problem.cloak_to_all_shifts(cloak_shifts)
        )

    def setup_edge_length_constraints(self, min_edge_length):
        def edge_length_constraints(horizontal_vertical_shifts):
            edge_lengths = compute_edge_lengths(
                self.forward_problem.cloaked_geometry.centroid_node_vectors(
                    *horizontal_vertical_shifts
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
            self.setup_angle_constraints(min_void_angle, min_block_angle)
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
                * jnp.ones((4 * len(self.forward_problem._bond_connectivity_cg),)),
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
