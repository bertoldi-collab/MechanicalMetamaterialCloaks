import jax
from mechanicalmetamaterialcloaks.utils import (
    SolutionType,
    SolutionData,
    ControlParams,
    GeometricalParams,
    MechanicalParams,
    LigamentParams,
    ContactParams,
)
from mechanicalmetamaterialcloaks.geometry import (
    QuadGeometry,
    compute_edge_angles,
    compute_edge_lengths,
    removed_blocks_in_2DLatticeGeometry,
    compute_inertia,
)
from mechanicalmetamaterialcloaks.energy import (
    build_strain_energy,
    ligament_energy,
    ligament_energy_linearized,
    build_contact_energy,
    combine_block_energies,
    compute_ligament_strains_history,
)
from mechanicalmetamaterialcloaks.dynamics import setup_dynamic_solver
from typing import Any, Optional, List, Union, Tuple, Dict
import dataclasses
from dataclasses import dataclass

import nlopt
import jax.numpy as jnp
from jax import flatten_util, pmap, random
from jax import value_and_grad, jit, vmap, jacobian
import matplotlib.pyplot as plt


@dataclass
class ForwardInput:
    """
    Input params for the forward solve function.
    As we will be using pmap, each loading param needs to be a tuple of length equal.

    Args:
        horizontal_shifts (ndarray): initial guess horizontal shifts.
        vertical_shifts (ndarray): initial guess vertical shifts.
        amplitude (Tuple[Any, ...]): amplitude of the dynamic loading.
        loading_rate (Tuple[Any, ...]): loading rate of the dynamic loading.
    """

    # Geometry
    horizontal_shifts: Any  # initial guess horizontal shifts
    vertical_shifts: Any  # initial guess vertical shifts

    # Dynamic loading
    amplitude: Tuple[Any, ...]
    loading_rate: Tuple[Any, ...]
    loading_angle: Tuple[Any, ...]


@dataclass
class ForwardProblem:
    """
    Forward problem class multi-tasking to design a shielding cloak for multiple loads.

    BCs:
        - One block at the centre of the grid is driven harmonically,
            with a sinusoidal translation along an axis defined with a loading angle.
        - The blocks at the four corner can be either clamped or left free to move.

    """

    # QuadGeometry
    n1_blocks: int
    n2_blocks: int
    spacing: float
    bond_length: float
    shielding_cloak_limits: List
    excited_blocks_limits: List

    # Mechanical
    k_stretch: float
    k_shear: float
    k_rot: float
    density: float
    damping: jnp.ndarray

    # Contact
    k_contact: float
    min_angle: float
    cutoff_angle: Any

    # Dynamic loading
    clamping_corners: bool

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
    name: str = "quads_dynamic_shielding_cloak_multi_loading"

    # Solution or list of solutions
    solution_data: Optional[Union[SolutionType, List[SolutionType]]] = None

    # Flag indicating that solve method is not available. It needs to be set up by calling self.setup().
    is_setup: bool = False

    def setup(self, excited_blocks_fn=None) -> None:
        """
        Set up forward solver.
        """

        # First, compute the dynamics with no loading.
        print("Setup intact case dynamics")
        # Intact geometry object contains attributes and methods to characterized the parametrization of nodes and bounds
        mother_geometry = QuadGeometry(
            n1_blocks=self.n1_blocks,
            n2_blocks=self.n2_blocks,
            spacing=self.spacing,
            bond_length=self.bond_length,
        )

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
                "At least matrix of nodes' shifts or an initial angle should be initialized."
            )

        # initialize connections between quads
        _bond_connectivity_mg = bond_connectivity_mg()
        _reference_bond_vectors_mg = reference_bond_vectors_mg()

        # Initial conditions
        self.state0 = jnp.array(
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
        damped_blocks = jnp.arange(mother_geometry.n_blocks)
        self.damping_matrix = self.damping * jnp.ones((mother_geometry.n_blocks, 3))

        # Dynamic input and BCs
        # Clamp the four corners
        constrained_block_DOF_pairs_mg = jnp.array(
            [
                jnp.tile(
                    jnp.array(
                        [
                            0,
                            self.n2_blocks - 1,
                            (self.n2_blocks - 1) * self.n1_blocks,
                            self.n2_blocks * self.n1_blocks - 1,
                        ]
                    ),
                    3,
                ),
                jnp.array([0] * 4 + [1] * 4 + [2] * 4),
            ]
        ).T

        # Make sure int type is respected
        constrained_block_DOF_pairs_mg = constrained_block_DOF_pairs_mg.astype(int)

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
                damping=self.damping_matrix,
                contact_params=ContactParams(
                    min_angle=self.min_angle,
                    cutoff_angle=self.cutoff_angle,
                    k_contact=self.k_contact,
                ),
            ),
            constraint_params=dict(),
        )

        # Setup solver
        solve_dynamics_MG = setup_dynamic_solver(
            geometry=mother_geometry,
            energy_fn=potential_energy_mg,
            constrained_block_DOF_pairs=constrained_block_DOF_pairs_mg,
            damped_blocks=damped_blocks,
            atol=1e-4,
        )

        # Analysis params
        simulation_time = self.simulation_time
        n_timepoints = self.n_timepoints
        timepoints = jnp.linspace(0, simulation_time, n_timepoints)

        # Solve dynamics of the mother geometry
        solution_mg = solve_dynamics_MG(
            state0=self.state0, timepoints=timepoints, control_params=control_params_mg
        )

        # Save data
        solutionData_mg = SolutionData(
            block_centroids=_init_block_centroids_mg,
            centroid_node_vectors=_init_centroid_node_vectors_mg,
            bond_connectivity=_bond_connectivity_mg,
            timepoints=timepoints,
            fields=solution_mg,
        )
        self.solutionData_mg = solutionData_mg

        # Build the solver dynamics of the shielding source
        print("Setup the solver of the shielding cloak dynamics")
        # Define cloak area
        cloaked_geometry = QuadGeometry(
            n1_blocks=self.n1_blocks,
            n2_blocks=self.n2_blocks,
            spacing=self.spacing,
            bond_length=self.bond_length,
        )

        self.cloaked_geometry = cloaked_geometry

        # Determine the localization of the shielding cloak
        # mgIDs_shielding_cloak_blocks contains ids of cloak's block: shield + excited block
        (
            self.mgIDs_shielding_cloak_blocks,
            self.mgIDs_surronding_area,
            self.grid_convert_mgIDs_to_cgIDs,
        ) = removed_blocks_in_2DLatticeGeometry(
            _init_block_centroids_mg, self.shielding_cloak_limits, mother_geometry
        )
        # Determine the localisation of the excited blocks
        self.mgIDs_excited_blocks, _, _ = removed_blocks_in_2DLatticeGeometry(
            _init_block_centroids_mg, self.excited_blocks_limits, mother_geometry
        )
        # Determine the localization of the blocks that will be optimized
        self.mgIDs_blocks_to_optimized = jnp.setdiff1d(
            self.mgIDs_shielding_cloak_blocks, self.mgIDs_excited_blocks
        )

        # Initialize the position of the centroids and the nodes
        (
            block_centroids_ss,
            centroid_node_vectors_ss,
            bond_connectivity_ss,
            reference_bond_vectors_ss,
        ) = cloaked_geometry.get_parametrization()

        self._bond_connectivity_ss = bond_connectivity_ss()
        self._reference_bond_vectors_ss = reference_bond_vectors_ss()

        # Block_centroids_ss and centroid_node_vectors_ss are functions of the nodes' shift
        self.block_centroids_ss = block_centroids_ss
        self.centroid_node_vectors_ss = centroid_node_vectors_ss

        # Dynamic input and BCs
        # Clamp the four corners
        if self.clamping_corners:
            clamped_blocks_DOF_pairs_cg = jnp.array(
                [
                    jnp.tile(
                        jnp.array(
                            [
                                0,
                                self.n2_blocks - 1,
                                (self.n2_blocks - 1) * self.n1_blocks,
                                self.n2_blocks * self.n1_blocks - 1,
                            ]
                        ),
                        3,
                    ),
                    jnp.array([0] * 4 + [1] * 4 + [2] * 4),
                ]
            ).T

        # Setup the source blocks
        self.nb_excited_blocks = self.mgIDs_excited_blocks.shape[0]

        source_blocks_DOF_pairs_cg = jnp.array(
            [
                jnp.tile(self.mgIDs_excited_blocks, 3),
                jnp.array(
                    [0] * self.nb_excited_blocks
                    + [1] * self.nb_excited_blocks
                    + [2] * self.nb_excited_blocks
                ),
            ]
        ).T

        # Make that int type is respected
        if self.clamping_corners:
            constrained_block_DOF_pairs_cg = jnp.concatenate(
                [source_blocks_DOF_pairs_cg, clamped_blocks_DOF_pairs_cg]
            ).astype(int)
        else:
            constrained_block_DOF_pairs_cg = source_blocks_DOF_pairs_cg.astype(int)

        # Dynamic loading
        # Apply the loading on the driven blocks
        def harmonic_ramp(t, amplitude, loading_rate):
            return amplitude * jnp.sin(2 * jnp.pi * loading_rate * t)

        if excited_blocks_fn is None:
            # If no function is provided, use the harmonic signal
            def constrained_DOFs_fn_cg(t, amplitude, loading_rate, loading_angle):
                fn = jnp.zeros((len(constrained_block_DOF_pairs_cg),))
                fn = fn.at[: self.nb_excited_blocks].set(
                    harmonic_ramp(t, amplitude * jnp.cos(loading_angle), loading_rate)
                )
                fn = fn.at[self.nb_excited_blocks: 2 * self.nb_excited_blocks].set(
                    harmonic_ramp(t, amplitude * jnp.sin(loading_angle), loading_rate)
                )
                return fn
        else:
            # Use the provided function e.g. when simulating with an experimental excitation
            def constrained_DOFs_fn_cg(t, *args, **kwargs):
                fn = jnp.zeros((len(constrained_block_DOF_pairs_cg),))
                fn = fn.at[: 3 * self.nb_excited_blocks].set(excited_blocks_fn(t))
                return fn

        # Construct energy of the system
        strain_energy_cg = build_strain_energy(
            bond_connectivity=self._bond_connectivity_ss,
            bond_energy_fn=ligament_energy_linearized
            if self.linearized_strains
            else ligament_energy,
        )
        contact_energy_cg = build_contact_energy(
            bond_connectivity=self._bond_connectivity_ss
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
            damped_blocks=damped_blocks,
            atol=1e-4,
        )

        def plot_sketch():
            # Plots to visualize region of the cloak and void
            blocks_for_plot = self.block_centroids_ss(
                *self.horizontal_vertical_shifts_mg
            )
            fig, axes = plt.subplots(constrained_layout=True)
            axes.plot(
                [
                    self.shielding_cloak_limits[k][0]
                    for k in range(len(self.shielding_cloak_limits))
                ],
                [
                    self.shielding_cloak_limits[k][1]
                    for k in range(len(self.shielding_cloak_limits))
                ],
                c="black",
            )
            axes.plot(
                [
                    self.excited_blocks_limits[k][0]
                    for k in range(len(self.excited_blocks_limits))
                ],
                [
                    self.excited_blocks_limits[k][1]
                    for k in range(len(self.excited_blocks_limits))
                ],
                c="blue",
            )
            axes.scatter(
                blocks_for_plot[self.mgIDs_excited_blocks, 0],
                blocks_for_plot[self.mgIDs_excited_blocks, 1],
                c="yellow",
            )
            axes.scatter(
                blocks_for_plot[self.mgIDs_blocks_to_optimized, 0],
                blocks_for_plot[self.mgIDs_blocks_to_optimized, 1],
                c="red",
            )
            axes.scatter(
                blocks_for_plot[self.mgIDs_surronding_area, 0],
                blocks_for_plot[self.mgIDs_surronding_area, 1],
                c="green",
            )
            axes.axis("equal")
            return fig, axes

        self.plot_sketch = plot_sketch
        # Define the mask of the allowed nodes to be shifted
        mask_shield_horizontal_shifts, mask_shield_vertical_shifts = (
            self.cloaked_geometry.get_shift_mask(self.mgIDs_blocks_to_optimized)
        )
        (
            mask_shield_horizontal_shifts_excited_blocks,
            mask_shield_vertical_shifts_excited_blocks,
        ) = self.cloaked_geometry.get_shift_mask(self.mgIDs_excited_blocks)
        self.mask_shield_horizontal_shifts = jnp.logical_and(
            mask_shield_horizontal_shifts,
            jnp.logical_not(mask_shield_horizontal_shifts_excited_blocks),
        )
        self.mask_shield_vertical_shifts = jnp.logical_and(
            mask_shield_vertical_shifts,
            jnp.logical_not(mask_shield_vertical_shifts_excited_blocks),
        )

        # Utility functions to map between all (mother) and cloak shifts
        def all_to_cloak_shifts(all_shifts):
            horizontal_shifts, vertical_shifts = all_shifts
            return horizontal_shifts[
                self.mask_shield_horizontal_shifts
            ], vertical_shifts[self.mask_shield_vertical_shifts]

        def cloak_to_all_shifts(cloak_shifts):
            horizontal_shifts, vertical_shifts = self.horizontal_vertical_shifts_mg
            horizontal_shifts_cloak_area, vertical_shifts_cloak_area = cloak_shifts
            horizontal_shifts = horizontal_shifts.at[
                self.mask_shield_horizontal_shifts
            ].set(horizontal_shifts_cloak_area)
            vertical_shifts = vertical_shifts.at[self.mask_shield_vertical_shifts].set(
                vertical_shifts_cloak_area
            )
            return horizontal_shifts, vertical_shifts

        def forward_all_shifts(
            horizontal_vertical_shifts: Tuple[
                jnp.ndarray, jnp.ndarray
            ],  # Design variables
            # Other forward inputs
            amplitude: float,
            loading_rate: float,
            loading_angle: float,
        ) -> SolutionData:
            # Design variables
            horizontal_shifts, vertical_shifts = horizontal_vertical_shifts

            constraint_params = dict(
                amplitude=amplitude,
                loading_rate=loading_rate,
                loading_angle=loading_angle,
            )

            # Change the control params given the new geometry
            _block_centroids_ss = self.block_centroids_ss(
                horizontal_shifts, vertical_shifts
            )
            _centroid_node_vectors_ss = self.centroid_node_vectors_ss(
                horizontal_shifts, vertical_shifts
            )

            # Control Parameters for Dynamic Solver

            control_params_cg = ControlParams(
                geometrical_params=GeometricalParams(
                    block_centroids=_block_centroids_ss,
                    centroid_node_vectors=_centroid_node_vectors_ss,
                ),
                mechanical_params=MechanicalParams(
                    bond_params=LigamentParams(
                        k_stretch=self.k_stretch,
                        k_shear=self.k_shear,
                        k_rot=self.k_rot,
                        reference_vector=self._reference_bond_vectors_ss,
                    ),
                    density=self.density,
                    damping=self.damping_matrix,
                    contact_params=ContactParams(
                        min_angle=self.min_angle,
                        cutoff_angle=self.cutoff_angle,
                        k_contact=self.k_contact,
                    ),
                ),
                constraint_params=constraint_params,
            )

            # Solve dynamics
            solution = solve_dynamics_cg(
                state0=self.state0,
                timepoints=timepoints,
                control_params=control_params_cg,
            )

            return SolutionData(
                block_centroids=_block_centroids_ss,
                centroid_node_vectors=_centroid_node_vectors_ss,
                bond_connectivity=self._bond_connectivity_ss,
                timepoints=timepoints,
                fields=solution,
            )

        # Setup forward
        def forward(
            horizontal_vertical_shifts_cloak_area: Tuple[
                jnp.ndarray, jnp.ndarray
            ],  # Design variables
            # Other forward inputs
            amplitude: float,
            loading_rate: float,
            loading_angle: float,
        ) -> SolutionData:
            horizontal_shifts, vertical_shifts = cloak_to_all_shifts(
                horizontal_vertical_shifts_cloak_area
            )
            return forward_all_shifts(
                (horizontal_shifts, vertical_shifts),
                amplitude,
                loading_rate,
                loading_angle,
            )

        # Used for cloak optimization
        self.solve = forward
        self.solve_all_shifts = forward_all_shifts  # Used for other analysis
        self.is_setup = True
        self.all_to_cloak_shifts = all_to_cloak_shifts
        self.cloak_to_all_shifts = cloak_to_all_shifts
        self.moving_blocks_ids = jnp.setdiff1d(
            jnp.arange(mother_geometry.n_blocks),
            jnp.unique(constrained_block_DOF_pairs_mg[:, 0]),
        )

    def compute_response_data(
        self, solution_data: Optional[SolutionData] = None
    ) -> dict:
        """Compute the response data associated with the solution data.

        This function computes the response data, such as the strains and the strain energy. If the solution data is not
        provided, the solution data stored in the class is used.
        The fields of the returned dictionary are:
            - all the fields of the SolutionData namedtuple
            - strain_energy_stretch: the strain energy due to stretching format as (n_timepoints, n_bonds).
            - strain_energy_shear: the strain energy due to shearing format as (n_timepoints, n_bonds).
            - strain_energy_bending: the strain energy due to bending format as (n_timepoints, n_bonds).
            - kinetic_energy: the kinetic energy format as (n_timepoints, n_blocks).

        Raises:
            ValueError: If the solution data is not available.
            ValueError: If the solution data is not of type SolutionData.

        Returns:
            dict: Dictionary with the response data.
        """

        if not self.is_setup:
            self.setup()

        if solution_data is None:
            if self.solution_data is None:
                raise ValueError("No solution data available!")
            else:
                solution_data = self.solution_data

        if type(solution_data) is not SolutionData:
            raise ValueError("Solution data is not of type SolutionData!")

        # Return a dictionary with the solution info
        dict_out = solution_data._asdict()
        # Compute strains
        axial_strain, shear_strain, bending_strain = compute_ligament_strains_history(
            solution_data.fields[:, 0],
            solution_data.centroid_node_vectors,
            solution_data.bond_connectivity,
            self.cloaked_geometry._reference_bond_vectors_ss(),
        )
        dict_out["strain_energy_stretch"] = (
            0.5 * self.k_stretch * (axial_strain * self.bond_length) ** 2
        )
        dict_out["strain_energy_shear"] = (
            0.5 * self.k_shear * (shear_strain * self.bond_length) ** 2
        )
        dict_out["strain_energy_bending"] = 0.5 * self.k_rot * bending_strain**2
        # Add kinetic energy to the dictionary
        inertia = compute_inertia(solution_data.centroid_node_vectors, self.density)
        dict_out["kinetic_energy"] = jnp.sum(
            0.5 * solution_data.fields[:, 1] ** 2 * inertia, axis=-1
        )

        return dict_out

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
    Optimization problem class for designing a multi loading shielding cloak to avoid perturbation in an area surronding a loading source, for multiple types of load.
    Objective is a weighted combination of the L2-norm of the displacement field in the area, surronding the cloak, of each forward problem.
    ForwardInput is used to define the set of forward problems.
    """

    # Forward problem provides a forward function that can be called on design variables and compressive strain
    forward_problem: ForwardProblem
    forward_input: ForwardInput
    weights: Tuple[float, ...]
    objective_values: Optional[List[Any]] = None
    objective_values_individual: Optional[List[Any]] = None
    design_values: Optional[List[Any]] = None
    constraints_violation: Optional[Dict[str, List[Any]]] = None
    name: str = ForwardProblem.name

    # Flag indicating that objective_fn method is not available. It needs to be set up by calling self.setup_objective().
    is_setup: bool = False

    def __post_init__(self):
        self.objective_values = (
            [] if self.objective_values is None else self.objective_values
        )
        self.objective_values_individual = (
            []
            if self.objective_values_individual is None
            else self.objective_values_individual
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

        # Retrieve geometry and density
        geometry = self.forward_problem.cloaked_geometry
        density = self.forward_problem.density

        forward_input_array = jnp.array(
            [
                self.forward_input.amplitude,
                self.forward_input.loading_rate,
                self.forward_input.loading_angle,
            ]
        ).T

        self.dimension_less = jnp.array(
            [
                1 / self.forward_problem.cloaked_geometry.spacing,
                1 / self.forward_problem.cloaked_geometry.spacing,
                1,
            ]
        )

        def delta_fn(
            horizontal_vertical_shifts: Tuple[jnp.ndarray, jnp.ndarray],
            forward_input: Tuple[float, ...],
        ):
            solution_data_cg = self.forward_problem.solve(
                horizontal_vertical_shifts, *forward_input
            )
            return (
                (
                    solution_data_cg.fields[
                        :, 0, self.forward_problem.mgIDs_surronding_area, :
                    ]
                    * self.dimension_less
                )
                ** 2
            ).sum()

        delta_fn_mapped = pmap(delta_fn, in_axes=(None, 0))

        # Total objective is the weighted sum of the delta functions for each forward problem
        def total_objective(
            horizontal_vertical_shifts: Tuple[jnp.ndarray, jnp.ndarray],
        ):
            return jnp.array(self.weights) @ delta_fn_mapped(
                horizontal_vertical_shifts, forward_input_array
            )

        self.objective_fn = total_objective
        self.objective_fn_individual = (
            lambda horizontal_vertical_shifts: delta_fn_mapped(
                horizontal_vertical_shifts, forward_input_array
            )
        )
        self.is_setup = True

    def setup_angle_constraints(self, min_void_angle=0.0, min_block_angle=0.0):
        centroid_node_vectors = (
            self.forward_problem.cloaked_geometry.centroid_node_vectors
        )
        bond_connectivity = self.forward_problem.cloaked_geometry.bond_connectivity()

        def angle_constraints(
            horizontal_vertical_shifts: Tuple[jnp.ndarray, jnp.ndarray],
        ):
            node_vectors = centroid_node_vectors(*horizontal_vertical_shifts)
            void_angles_1, void_angles_2, block_angles_1, block_angles_2 = vmap(
                lambda b: jnp.mod(
                    jnp.array(compute_edge_angles(node_vectors, b)), 2 * jnp.pi
                ),
                in_axes=0,
            )(bond_connectivity).T

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
        centroid_node_vectors = (
            self.forward_problem.cloaked_geometry.centroid_node_vectors
        )

        def edge_length_constraints(
            horizontal_vertical_shifts: Tuple[jnp.ndarray, jnp.ndarray],
        ):
            edge_lengths = compute_edge_lengths(
                centroid_node_vectors(*horizontal_vertical_shifts)
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

        # objective_and_grad = jit(value_and_grad(self.objective_fn))
        objective_and_grad = value_and_grad(self.objective_fn)

        def nlopt_objective(x, grad):
            v, g = objective_and_grad(unflatten(x))  # jax evaluation
            vs = self.objective_fn_individual(unflatten(x))
            self.objective_values.append(v)
            self.objective_values_individual.append(vs)
            self.design_values.append(unflatten(x))

            print(
                f"Iteration: {len(self.objective_values)}\nObjective = {self.objective_values[-1]}\nObjectives = {self.objective_values_individual[-1]}"
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
                    (
                        4
                        * len(
                            self.forward_problem.cloaked_geometry.bond_connectivity()
                        ),
                    )
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

        # Run optimization
        opt.optimize(initial_guess_flattened)

        # Store forward solution data for the last design
        self.compute_best_forwards()

    def compute_best_forwards(
        self, compute_again: bool = False, n_timepoints: int = 200
    ):
        if (not compute_again) and len(self.design_values) == 0:
            raise ValueError("No design has been optimized yet.")

        if not self.forward_problem.is_setup:
            self.forward_problem.setup()

        self.forward_problem.solution_data = {}
        self.forward_problem.solution_data["mg"] = self.forward_problem.solutionData_mg

        forward_input_array = jnp.array(
            [
                self.forward_input.amplitude,
                self.forward_input.loading_rate,
                self.forward_input.loading_angle,
            ]
        ).T

        self.forward_problem.solution_data["opcg"] = [
            self.forward_problem.solve(
                self.design_values[-1],
                *forward_input,
            )
            for forward_input in forward_input_array
        ]

        self.forward_problem.solution_data["ig"] = [
            self.forward_problem.solve(
                self.design_values[0],
                *forward_input,
            )
            for forward_input in forward_input_array
        ]

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
        optimization_data.forward_problem.horizontal_vertical_shifts_mg = jax.tree.map(
            lambda x: jnp.array(x),
            optimization_data.forward_problem.horizontal_vertical_shifts_mg,
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
