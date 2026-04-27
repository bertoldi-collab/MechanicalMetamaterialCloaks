import dataclasses
import jax
import matplotlib.pyplot as plt
import jax.numpy as jnp
from jax import grad, jacobian, jit, value_and_grad, vmap, flatten_util
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
    QuadGeometry,
    compute_edge_angles,
    compute_edge_lengths,
    removed_blocks_in_2DLatticeGeometry,
    target_cloak_area_in_2DLatticeGeometry,
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
    Forward problem for the static cloaking of a rigid inclusion in a quad metamaterial.
    BCs:
        - Left column clamped.
        - Right column slowly driven in compression.
    """

    # QuadGeometry
    n1_blocks: int
    n2_blocks: int
    spacing: Any
    bond_length: Any

    # Cloak Geometry
    void: List
    width_strip_cloak_area: float

    # Mechanical
    k_stretch: Any
    k_shear: Any
    k_rot: Any
    inclusion_stiffness_factor: Any
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
    horizontal_vertical_shifts: Any = None  # one of the two argument needs to be fill
    initial_angle: Optional[float] = None

    # Problem name
    name: str = "quads_static_cloaking_rigid_inclusion"

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

        # Geometry
        geometry = QuadGeometry(
            n1_blocks=self.n1_blocks,
            n2_blocks=self.n2_blocks,
            spacing=self.spacing,
            bond_length=self.bond_length,
        )
        (
            block_centroids,
            centroid_node_vectors,
            bond_connectivity,
            reference_bond_vectors,
        ) = geometry.get_parametrization()
        if self.initial_angle == None:
            _init_centroid_node_vectors = centroid_node_vectors(
                *self.horizontal_vertical_shifts
            )
            _init_block_centroids = block_centroids(*self.horizontal_vertical_shifts)
        else:
            self.horizontal_vertical_shifts = geometry.get_design_from_rotated_square(
                angle=self.initial_angle
            )
            _init_centroid_node_vectors = centroid_node_vectors(
                *self.horizontal_vertical_shifts
            )
            _init_block_centroids = block_centroids(*self.horizontal_vertical_shifts)
        _bond_connectivity = bond_connectivity()
        _reference_bond_vectors = reference_bond_vectors()
        self.geometry = geometry

        # Initial conditions
        state0 = jnp.array(
            [
                jnp.zeros((geometry.n_blocks, 3)),  # Initial position
                jnp.zeros((geometry.n_blocks, 3)),  # Initial velocity
            ]
        )

        # Damping
        damped_blocks = jnp.arange(geometry.n_blocks)
        self.damping_matrix = self.damping * jnp.ones((geometry.n_blocks, 3))

        # Dynamic input and BCs
        driven_block_DOF_pairs = jnp.array(
            [
                jnp.tile(
                    jnp.arange(0, geometry.n2_blocks) * geometry.n1_blocks
                    + geometry.n1_blocks
                    - 1,
                    3,
                ),
                jnp.array(
                    [0] * self.n2_blocks + [1] * self.n2_blocks + [2] * self.n2_blocks
                ),
            ]
        ).T
        clamped_blocks = jnp.array(
            [
                jnp.tile(
                    jnp.arange(
                        (self.n2_blocks / 2 - self.n2_blocks / 2),
                        self.n2_blocks / 2 + self.n2_blocks / 2,
                    ).astype(int)
                    * geometry.n1_blocks,
                    3,
                ),
                jnp.array(
                    [0] * self.n2_blocks + [1] * self.n2_blocks + [2] * self.n2_blocks
                ),
            ]
        ).T
        self.driven_block_DOF_pairs = driven_block_DOF_pairs

        constrained_block_DOF_pairs = jnp.concatenate(
            [driven_block_DOF_pairs, clamped_blocks]
        ).astype(int)

        moving_blocks_ids = jnp.array(
            [
                j + i * self.n1_blocks
                for i in range(self.n2_blocks)
                for j in range(1, self.n1_blocks - 1)
            ]
        )
        self.moving_blocks_ids = moving_blocks_ids

        # Dynamic loading
        loading_vector = jnp.zeros((len(constrained_block_DOF_pairs),))
        loading_vector = loading_vector.at[: self.n2_blocks].set(-1.0)

        def applied_displacement(t, amplitude, simulation_time):
            return amplitude * (1 + jnp.tanh(6 * (2 * t / simulation_time - 1))) / 2

        self.applied_displacement = applied_displacement

        def constrained_DOFs_fn(t, amplitude, simulation_time):
            return loading_vector * applied_displacement(t, amplitude, simulation_time)

        # Construct strain energy
        strain_energy = build_strain_energy(
            bond_connectivity=_bond_connectivity,
            bond_energy_fn=ligament_energy_linearized
            if self.linearized_strains
            else ligament_energy,
        )
        contact_energy = build_contact_energy(bond_connectivity=_bond_connectivity)
        potential_energy = combine_block_energies(strain_energy, contact_energy)
        self.elastic_forces = grad(potential_energy)

        # Solution of the intact geometry
        # Control Parameters for Dynamic Solver
        control_params_intact = ControlParams(
            geometrical_params=GeometricalParams(
                block_centroids=_init_block_centroids,
                centroid_node_vectors=_init_centroid_node_vectors,
            ),
            mechanical_params=MechanicalParams(
                bond_params=LigamentParams(
                    k_stretch=self.k_stretch,
                    k_shear=self.k_shear,
                    k_rot=self.k_rot,
                    reference_vector=_reference_bond_vectors,
                ),
                density=self.density,
                damping=self.damping_matrix,
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
        self.control_params_intact = control_params_intact

        # Setup solver
        solve_dynamics = setup_dynamic_solver(
            geometry=geometry,
            energy_fn=potential_energy,
            constrained_block_DOF_pairs=constrained_block_DOF_pairs,
            constrained_DOFs_fn=constrained_DOFs_fn,
            damped_blocks=damped_blocks,
            atol=self.atol,
            rtol=self.rtol,
        )
        # Analysis params
        simulation_time = self.simulation_time
        n_timepoints = self.n_timepoints
        timepoints = jnp.linspace(0, simulation_time, n_timepoints)
        # Solve dynamics
        solution_intact = solve_dynamics(
            state0=state0, timepoints=timepoints, control_params=control_params_intact
        )

        print("compute solution MG done")

        # Save data
        solution_data_intact = SolutionData(
            block_centroids=_init_block_centroids,
            centroid_node_vectors=_init_centroid_node_vectors,
            bond_connectivity=_bond_connectivity,
            timepoints=timepoints,
            fields=solution_intact,
        )
        self.solution_data_intact_fields = solution_data_intact.fields
        self.solution_data_intact = solution_data_intact

        # Cloak Geometry
        # Defined IDs of the inclusion, the cloak area, and the surrounding area
        inclusion_IDs, cloak_and_surronding_IDs, _ = (
            removed_blocks_in_2DLatticeGeometry(
                initial_block_centroids_mother_geometry=block_centroids(
                    *self.horizontal_vertical_shifts
                ),
                void=self.void,
                mother_geometry=geometry,
            )
        )
        cloak_IDs = target_cloak_area_in_2DLatticeGeometry(
            initial_block_centroids_mother_geometry=block_centroids(
                *self.horizontal_vertical_shifts
            ),
            void=self.void,
            id_kept_blocks=cloak_and_surronding_IDs,
            middle_cloak_area=self.middle_cloak_area,
            width_strip_cloak_area=self.width_strip_cloak_area,
        )
        self.inclusion_IDs = inclusion_IDs
        self.cloak_IDs = cloak_IDs
        self.surrounding_IDs = jnp.array(
            [
                i
                for i in range(geometry.n_blocks)
                if i not in inclusion_IDs and i not in cloak_IDs
            ]
        )

        def plot_sketch():
            # Plots to visualize region of the cloak and void
            blocks_for_plot = block_centroids(*self.horizontal_vertical_shifts)
            fig, axes = plt.subplots(constrained_layout=True)
            axes.plot(*jnp.array(self.void).T)
            axes.scatter(
                blocks_for_plot[self.cloak_IDs, 0],
                blocks_for_plot[self.cloak_IDs, 1],
                c="red",
            )
            axes.scatter(
                blocks_for_plot[self.inclusion_IDs, 0],
                blocks_for_plot[self.inclusion_IDs, 1],
                c="black",
            )
            axes.scatter(
                blocks_for_plot[self.surrounding_IDs, 0],
                blocks_for_plot[self.surrounding_IDs, 1],
                c="green",
            )
            axes.axis("equal")
            return fig, axes

        # Mask for the cloak shifts
        # FIXME: This somehow it is not correct as it include the shifts inside the inclusion
        mask_horizontal_shift_cloak, mask_vertical_shift_cloak = (
            geometry.get_shift_mask(self.cloak_IDs)
        )
        self.mask_horizontal_shift_cloak = mask_horizontal_shift_cloak
        self.mask_vertical_shift_cloak = mask_vertical_shift_cloak
        # Cloak shifts from self.horizontal_vertical_shifts (can be used as initial guess for the optimization)
        self.horizontal_vertical_shifts_cloak = (
            self.horizontal_vertical_shifts[0][mask_horizontal_shift_cloak],
            self.horizontal_vertical_shifts[1][mask_vertical_shift_cloak],
        )

        # Utility functions to map between all (mother) and cloak shifts
        def all_to_cloak_shifts(all_shifts):
            horizontal_shifts, vertical_shifts = all_shifts
            return horizontal_shifts[self.mask_horizontal_shift_cloak], vertical_shifts[
                self.mask_vertical_shift_cloak
            ]

        def cloak_to_all_shifts(cloak_shifts):
            horizontal_shifts, vertical_shifts = self.horizontal_vertical_shifts
            horizontal_shifts = horizontal_shifts.at[
                self.mask_horizontal_shift_cloak
            ].set(cloak_shifts[0])
            vertical_shifts = vertical_shifts.at[self.mask_vertical_shift_cloak].set(
                cloak_shifts[1]
            )
            return horizontal_shifts, vertical_shifts

        # Mask for the bonds in the inclusion
        inclusion_nodes = jnp.array(
            [
                jnp.arange(geometry.n_npb * block, geometry.n_npb * (block + 1))
                for block in self.inclusion_IDs
            ]
        ).flatten()
        # Target bonds
        inclusion_bonds = jnp.isin(_bond_connectivity, inclusion_nodes).any(axis=-1)
        inclusion_bonds_stiffness_factor = jnp.ones(_bond_connectivity.shape[0])
        inclusion_bonds_stiffness_factor = inclusion_bonds_stiffness_factor.at[
            inclusion_bonds
        ].set(self.inclusion_stiffness_factor)

        def control_params_fn(horizontal_vertical_shifts_cloak_area):
            horizontal_shifts, vertical_shifts = cloak_to_all_shifts(
                horizontal_vertical_shifts_cloak_area
            )

            # Change the control params given the new geometry
            _block_centroids_cloak = block_centroids(horizontal_shifts, vertical_shifts)
            _centroid_node_vectors_cloak = centroid_node_vectors(
                horizontal_shifts, vertical_shifts
            )

            control_params_cloak = ControlParams(
                geometrical_params=GeometricalParams(
                    block_centroids=_block_centroids_cloak,
                    centroid_node_vectors=_centroid_node_vectors_cloak,
                ),
                mechanical_params=MechanicalParams(
                    bond_params=LigamentParams(
                        k_stretch=self.k_stretch * inclusion_bonds_stiffness_factor,
                        k_shear=self.k_shear * inclusion_bonds_stiffness_factor,
                        k_rot=self.k_rot * inclusion_bonds_stiffness_factor,
                        reference_vector=_reference_bond_vectors,
                    ),
                    density=self.density,
                    damping=self.damping_matrix,
                    contact_params=ContactParams(
                        min_angle=self.min_angle,
                        cutoff_angle=self.cutoff_angle,
                        k_contact=self.k_contact,
                    ),
                ),
                constraint_params=dict(
                    amplitude=self.amplitude,
                    simulation_time=self.simulation_time,
                ),
            )
            return control_params_cloak

        self.control_params_fn = control_params_fn

        # Setup forward
        def forward(horizontal_vertical_shifts_cloak_area):
            control_params = control_params_fn(horizontal_vertical_shifts_cloak_area)
            # Solve dynamics
            solution = solve_dynamics(
                state0=state0,
                timepoints=timepoints,
                control_params=control_params,
            )
            return SolutionData(
                block_centroids=control_params.geometrical_params.block_centroids,
                centroid_node_vectors=control_params.geometrical_params.centroid_node_vectors,
                bond_connectivity=_bond_connectivity,
                timepoints=timepoints,
                fields=solution,
            )

        self.solve = forward
        self.is_setup = True
        self.all_to_cloak_shifts = all_to_cloak_shifts
        self.cloak_to_all_shifts = cloak_to_all_shifts
        self.plot_sketch = plot_sketch
        print("setup forward done")

    def force_displacement(
        self, solution_data: SolutionData, control_params: ControlParams
    ):
        if self.is_setup:
            displacement_history = solution_data.fields[:, 0]
            block_DOF_pairs = self.driven_block_DOF_pairs[: self.n2_blocks]
            force_history = vmap(
                lambda u: jnp.sum(
                    self.elastic_forces(u, control_params)[
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

    def force_displacement_intact(
        self,
        solution_data: SolutionData,
    ):
        return self.force_displacement(solution_data, self.control_params_intact)

    def force_displacement_cloak(
        self, solution_data: SolutionData, horizontal_vertical_shifts_cloak
    ):
        return self.force_displacement(
            solution_data, self.control_params_fn(horizontal_vertical_shifts_cloak)
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
                dict_in["solution_data"] = {
                    key: SolutionData(
                        **solution) if type(solution) is not list else [SolutionData(**s) for s in solution]
                    for key, solution in dict_in["solution_data"].items()
                }
            elif type(dict_in["solution_data"]) is list:
                dict_in["solution_data"] = [
                    SolutionData(**solution) for solution in dict_in["solution_data"]
                ]
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
                1 / self.forward_problem.geometry.spacing,
                1 / self.forward_problem.geometry.spacing,
                1,
            ]
        )

        if self.objective_type == "integrated":
            self.max_delta_mg = (
                (
                    self.forward_problem.solution_data_intact_fields[
                        -1, 0, self.forward_problem.surrounding_IDs, :
                    ]
                    * self.dimension_less
                )
                ** 2
            ).sum() ** 0.5

            # Delta function integrated over time to match configuration along the entire time interval
            def delta_function_integrated(horizontal_vertical_shifts_cloak_area):
                """
                Args :
                    horizontal_vertical_shifts_cloak_area: tuple[jnp.ndarray, jnp.ndarray]
                        More precisely, if horizontal_shift, vertical_shift = horizontal_vertical_shifts_cloak_area
                        then     horizontal_shift(jnp.ndarray of shape(geometry_cloak.nb_horizontal_shifts,2))
                            and  vertical_shift((jnp.ndarray of shape(geometry_cloak.nb_vertical_shifts,2)))
                """
                # Solve forward
                solution_data_cg = self.forward_problem.solve(
                    horizontal_vertical_shifts_cloak_area
                )

                return (
                    (
                        (
                            (
                                solution_data_cg.fields[
                                    :, 0, self.forward_problem.surrounding_IDs, :
                                ]
                                - self.forward_problem.solution_data_intact_fields[
                                    :, 0, self.forward_problem.surrounding_IDs, :
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
                        self.forward_problem.solution_data_intact_fields[
                            :, 0, self.forward_problem.surrounding_IDs, :
                        ]
                        * self.dimension_less
                    )
                    ** 2
                ).sum(axis=(1, 2))
                ** 0.5
            )

            # Delta function at the final time to match final configuration
            def delta_function_final(horizontal_vertical_shifts_cloak_area):
                """
                Args :
                    horizontal_vertical_shifts_cloak_area: tuple[jnp.ndarray, jnp.ndarray]
                        More precisely, if horizontal_shift, vertical_shift = horizontal_vertical_shifts_cloak_area
                        then     horizontal_shift(jnp.ndarray of shape(geometry_cloak.nb_horizontal_shifts,2))
                            and  vertical_shift((jnp.ndarray of shape(geometry_cloak.nb_vertical_shifts,2)))
                """
                # Solve forward
                solution_data_cg = self.forward_problem.solve(
                    horizontal_vertical_shifts_cloak_area
                )

                return (
                    (
                        (
                            (
                                solution_data_cg.fields[
                                    -1, 0, self.forward_problem.surrounding_IDs, :
                                ]
                                - self.forward_problem.solution_data_intact_fields[
                                    -1, 0, self.forward_problem.surrounding_IDs, :
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

    def setup_angle_constraints(self, min_void_angle=0.0, min_block_angle=0.0):
        # centroid_node_vectors = self.forward_problem.geometry.centroid_node_vectors

        def angle_constraints(horizontal_vertical_shifts):
            _centroid_node_vectors = (
                self.forward_problem.geometry.centroid_node_vectors(
                    *horizontal_vertical_shifts
                )
            )
            _bond_connectivity = self.forward_problem.geometry.bond_connectivity()

            void_angles_1, void_angles_2, block_angles_1, block_angles_2 = vmap(
                lambda b: jnp.mod(
                    jnp.array(compute_edge_angles(_centroid_node_vectors, b)),
                    2 * jnp.pi,
                ),
                in_axes=0,
            )(_bond_connectivity).T

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
                self.forward_problem.geometry.centroid_node_vectors(
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
                * jnp.ones(
                    (4 * len(self.forward_problem.geometry.bond_connectivity()),)
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
                    self.forward_problem.geometry.n_blocks
                    * self.forward_problem.geometry.n_npb
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
            self.forward_problem.solution_data_intact
        )  # Add the mother geometry solution
        self.forward_problem.solution_data.append(
            self.forward_problem.solve(self.design_values[0])
        )  # Add the initial design solution
        self.forward_problem.solution_data.append(
            self.forward_problem.solve(self.design_values[-1])
        )  # Add the optimized design solution

        return self.forward_problem.solution_data

    @staticmethod
    def from_data(optimization_data):
        optimization_data.forward_problem = ForwardProblem.from_data(
            optimization_data.forward_problem
        )
        optimization_data.forward_input = ForwardInput(
            **optimization_data.forward_input
        )
        # Ensure design values are iterable of jax arrays
        optimization_data.design_values = jax.tree.map(
            lambda x: jnp.array(x), optimization_data.design_values
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
