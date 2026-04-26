import argparse
from multiprocessing import Pool
from pathlib import Path
from typing import Optional, Tuple, Union

import cv2
import matplotlib.animation as animation
import matplotlib.pyplot as plt
import numpy as np
from jax import vmap
from matplotlib import cm, colors
from matplotlib.collections import LineCollection, PatchCollection, PolyCollection
from matplotlib.colors import ListedColormap
from matplotlib.patches import Polygon

from mechanicalmetamaterialcloaks.geometry import (
    compute_xy_limits,
    current_coordinates,
    rotation_matrix,
)
from mechanicalmetamaterialcloaks.utils import EigenmodeData, SolutionData, load_data


def orange_blue_cmap():
    """
    Custom colormap
    """

    top = cm.get_cmap("Oranges_r", 128)
    bottom = cm.get_cmap("Blues", 128)
    newcolors = np.vstack((top(np.linspace(0, 1, 128)), bottom(np.linspace(0, 1, 128))))
    return ListedColormap(newcolors, name="OrangeBlue")


def plot_energy(dat):
    pot_energy = []
    kin_energy = []
    for i in range(dat.fields.shape[0]):
        dx = dat.fields[i, 0, :, 0]
        dy = dat.fields[i, 0, :, 1]

        pot_energy.append(np.sum(dx**2 + dy**2))
        vx = dat.fields[i, 1, :, 0]
        vy = dat.fields[i, 1, :, 1]
        kin_energy.append(np.sum(vx**2 + vy**2))

    plt.figure(2)
    plt.plot(dat.timepoints, kin_energy, lw=2, label="kinetic")
    plt.plot(dat.timepoints, pot_energy, lw=2, label="potential")
    plt.legend()
    plt.xlabel("Time")
    plt.ylabel("Energy")
    plt.savefig("out/energy.png", dpi=300, bbox_inches="tight")


def generate_polygons(
    block_centroids, centroid_node_vectors, block_displacements=None, deformed=False
):
    """
    docstring
    """

    if deformed and block_displacements is not None:
        polygons = [
            Polygon((rotation_matrix(DOFs[-1]) @ vertices.T).T + centroid + DOFs[:2])
            for vertices, centroid, DOFs in zip(
                centroid_node_vectors, block_centroids, block_displacements
            )
        ]
    else:
        polygons = [
            Polygon(vertices + centroid)
            for vertices, centroid in zip(centroid_node_vectors, block_centroids)
        ]

    return polygons


def generate_patch_collection(
    block_centroids,
    centroid_node_vectors,
    block_displacements=None,
    field_values=None,
    deformed=False,
    clim=None,
    cmap=orange_blue_cmap(),
):
    """
    docstring
    """

    polygons = generate_polygons(
        block_centroids,
        centroid_node_vectors,
        block_displacements=block_displacements,
        deformed=deformed,
    )
    patches = PatchCollection(polygons, cmap=cmap, alpha=0.95)
    if field_values is not None:
        patches.set_array(field_values)
        min_value, max_value = (
            (field_values.min(), field_values.max()) if clim is None else clim
        )
        patches.set_clim(min_value, max_value)
    patches.set(edgecolor="black", linewidth=0.5)

    return patches


def generate_bond_collection(
    block_centroids,
    centroid_node_vectors,
    bond_connectivity,
    block_displacements=None,
    deformed=False,
):
    """
    docstring
    """

    # Generate collection of bonds as lines
    if deformed and block_displacements is not None:
        block_coords = current_coordinates(
            centroid_node_vectors,
            block_centroids,
            block_displacements[:, -1],
            block_displacements[:, :2],
        )
    else:
        block_coords = vmap(
            lambda centroid, centroid_node_vector: centroid + centroid_node_vector,
            in_axes=(0, 0),
        )(centroid_node_vectors, block_centroids)

    n_blocks, n_npb, _ = block_coords.shape
    node_coords = block_coords.reshape((n_blocks * n_npb, 2))

    return LineCollection(node_coords[bond_connectivity], color="black", linewidth=0.5)


def plot_geometry(
    block_centroids,
    centroid_node_vectors,
    bond_connectivity,
    block_displacements=None,
    deformed=False,
    color="#2980b9",
    figsize=None,
    xlim=None,
    ylim=None,
    ax=None,
):
    """
    docstring
    """

    if ax is None:
        fig, ax = plt.subplots(figsize=figsize, constrained_layout=True)
        ax.axis("equal")
    # Generate collection of blocks as polygons
    patches = generate_patch_collection(
        block_centroids,
        centroid_node_vectors,
        block_displacements=block_displacements,
        deformed=deformed,
    )
    patches.set(color=color)
    patches.set(edgecolor="black", linewidth=0.5)
    ax.add_collection(patches)
    # Generate collection of bonds as lines
    collection_bonds = generate_bond_collection(
        block_centroids,
        centroid_node_vectors,
        bond_connectivity,
        block_displacements=block_displacements,
        deformed=deformed,
    )
    ax.add_collection(collection_bonds)

    if deformed and block_displacements is not None:
        points = current_coordinates(
            centroid_node_vectors,
            block_centroids,
            block_displacements[:, -1],
            block_displacements[:, :2],
        ).reshape((-1, 2))
    else:
        points = (block_centroids[:, None, :] + centroid_node_vectors).reshape((-1, 2))

    _xlim, _ylim = compute_xy_limits(points)
    xlim = _xlim if xlim is None else xlim
    ylim = _ylim if ylim is None else ylim
    ax.set(xlim=xlim, ylim=ylim)

    fig = ax.get_figure()

    return fig, ax


def compute_field_values(data: SolutionData, field):
    if field == "ux":
        field_values = data.fields[:, 0, :, 0]
    elif field == "uy":
        field_values = data.fields[:, 0, :, 1]
    elif field == "theta":
        field_values = data.fields[:, 0, :, 2]
    elif field == "vx":
        field_values = data.fields[:, 1, :, 0]
    elif field == "vy":
        field_values = data.fields[:, 1, :, 1]
    elif field == "omega":
        field_values = data.fields[:, 1, :, 2]
    elif field == "u":
        field_values = (
            data.fields[:, 0, :, 0] ** 2 + data.fields[:, 0, :, 1] ** 2
        ) ** 0.5
    elif field == "v":
        field_values = (
            data.fields[:, 1, :, 0] ** 2 + data.fields[:, 1, :, 1] ** 2
        ) ** 0.5
    elif field == "theta_abs":
        field_values = np.abs(data.fields[:, 0, :, 2])
    else:
        raise ValueError

    return field_values


def field_name_to_label(field):
    if field == "ux":
        return r"$u_1$"
    elif field == "uy":
        return r"$u_2$"
    elif field == "theta":
        return r"$\theta$"
    elif field == "vx":
        return r"$\dot{u}_1$"
    elif field == "vy":
        return r"$\dot{u}_2$"
    elif field == "omega":
        return r"$\dot{\theta}$"
    elif field == "u":
        return r"$u$"
    elif field == "v":
        return r"$\dot{u}$"
    elif field == "theta_abs":
        return r"$\lvert\theta\rvert$"
    else:
        return field


def plot_geometry_field_overlaid(
    data: SolutionData,
    field: str,
    timepoint: int,
    field_values: Optional[np.ndarray] = None,
    deformed: bool = False,
    colorbar: bool = True,
    figsize: Optional[Tuple[float, float]] = None,
    xlim: Optional[Tuple[float, float]] = None,
    ylim: Optional[Tuple[float, float]] = None,
    cmap=orange_blue_cmap(),
    norm=None,
    vlim: Optional[Tuple[float, float]] = None,
    legend_label: Optional[str] = None,
    fontsize: int = 14,
    ticksize: int = 14,
    axis: bool = True,
    ax=None,
):
    fig, axes = plot_geometry(
        data.block_centroids,
        data.centroid_node_vectors,
        data.bond_connectivity,
        block_displacements=data.fields[timepoint, 0, :, :] if deformed else None,
        deformed=deformed,
        figsize=figsize,
        xlim=xlim,
        ylim=ylim,
        ax=ax,
    )
    # Color the blocks according to the field values
    vmin, vmax = vlim if vlim is not None else (None, None)
    field_values_all_times = (
        compute_field_values(data, field) if field_values is None else field_values
    )
    field_values_min = field_values_all_times.min() if vmin is None else vmin
    field_values_max = field_values_all_times.max() if vmax is None else vmax
    _field_values = field_values_all_times[timepoint]

    axes.collections[0].set_array(_field_values)
    axes.collections[0].set_cmap(cmap)
    axes.collections[0].set_norm(
        colors.Normalize(vmin=field_values_min, vmax=field_values_max)
        if norm is None
        else norm
    )
    if colorbar:
        cb = fig.colorbar(axes.collections[0], pad=0.02, label=legend_label, aspect=40)
        cb.ax.tick_params(labelsize=ticksize)
        cb.ax.set_ylabel(legend_label, fontsize=fontsize)
    if not axis:
        axes.axis("off")
    axes.tick_params(labelsize=ticksize)

    return fig, axes


def prepare_solution_figure(
    data: SolutionData,
    field,
    frame_range,
    figsize,
    cmap=orange_blue_cmap(),
    vlim=None,
    legend_label=None,
    fontsize=14,
    ticksize=14,
    axis=True,
    field_values=None,
):
    field_values = (
        compute_field_values(data, field) if field_values is None else field_values
    )
    _legend_label = field_name_to_label(field)

    min_value, max_value = field_values.min(), field_values.max()
    vmin, vmax = vlim if vlim is not None else (min_value, max_value)
    _legend_label = legend_label if legend_label is not None else _legend_label

    fig, axes = plt.subplots(figsize=figsize, constrained_layout=True)
    axes.axis("equal")
    axes.tick_params(labelsize=ticksize)
    if not axis:
        axes.axis("off")
    cb = fig.colorbar(
        cm.ScalarMappable(cmap=cmap, norm=colors.Normalize(vmin=vmin, vmax=vmax)),
        pad=0.02,
        label=_legend_label,
        aspect=40,
    )
    cb.ax.tick_params(labelsize=ticksize)
    cb.ax.set_ylabel(_legend_label, fontsize=fontsize)
    frames = range(len(data.timepoints)) if frame_range is None else frame_range

    return field_values, min_value, max_value, fig, axes, frames


def prepare_solution_figure_for_several_animations(
    data: SolutionData,
    field,
    vlim=None,
    legend_label=None,
    fontsize=14,
    ticksize=14,
    axis=True,
    field_values=None,
):
    field_values = (
        compute_field_values(data, field) if field_values is None else field_values
    )
    _legend_label = field_name_to_label(field)

    min_value, max_value = field_values.min(), field_values.max()
    vmin, vmax = vlim if vlim is not None else (min_value, max_value)
    _legend_label = legend_label if legend_label is not None else _legend_label

    return field_values, min_value, max_value, vmin, vmax, _legend_label


def prepare_mode_figure(
    data: EigenmodeData,
    field,
    mode_range,
    figsize,
    cmap=orange_blue_cmap(),
    vlim=None,
    legend_label=None,
    fontsize=14,
    ticksize=14,
    axis=True,
):
    if field == "ux":
        field_values = data.fields[:, :, 0]
        _legend_label = r"$u_1$"
    elif field == "uy":
        field_values = data.fields[:, :, 1]
        _legend_label = r"$u_2$"
    elif field == "theta":
        field_values = data.fields[:, :, 2]
        _legend_label = r"$\theta$"
    elif field == "u":
        field_values = (data.fields[:, :, 0] ** 2 + data.fields[:, :, 1] ** 2) ** 0.5
        _legend_label = r"$u$"
    elif field == "theta_abs":
        field_values = np.abs(data.fields[:, :, 2])
        _legend_label = r"$\lvert\theta\rvert$"
    else:
        raise ValueError

    vmin, vmax = vlim if vlim is not None else (None, None)
    _legend_label = legend_label if legend_label is not None else _legend_label

    fig, axes = plt.subplots(figsize=figsize, constrained_layout=True)
    axes.axis("equal")
    axes.tick_params(labelsize=ticksize)
    if not axis:
        axes.axis("off")
    cb = fig.colorbar(
        cm.ScalarMappable(cmap=cmap, norm=colors.Normalize(vmin=vmin, vmax=vmax)),
        pad=0.02,
        label=_legend_label,
        aspect=40,
    )
    cb.ax.tick_params(labelsize=ticksize)
    cb.ax.set_ylabel(_legend_label, fontsize=fontsize)
    frames = range(len(data.fields)) if mode_range is None else mode_range

    return field_values, fig, axes, frames


def generate_mode_images(
    data: EigenmodeData,
    field,
    out_dir,
    deformed=False,
    mode_range=None,
    scale_deformation=1,
    figsize=None,
    xlim=None,
    ylim=None,
    dpi=200,
    geometry=None,
    mesh=None,
    cmap=orange_blue_cmap(),
    vlim=None,
    legend_label=None,
    fontsize=14,
    ticksize=14,
    axis=True,
):
    """
    mesh=None: if set to True, a mesh connecting the centroids of each block is superimposed on the images
    docstring
    """

    field_values, fig, axes, frames = prepare_mode_figure(
        data,
        field,
        mode_range,
        figsize,
        cmap=cmap,
        vlim=vlim,
        legend_label=legend_label,
        fontsize=fontsize,
        ticksize=ticksize,
        axis=axis,
    )
    block_centroids = data.block_centroids
    centroid_node_vectors = data.centroid_node_vectors
    block_displacements = data.fields

    for i in frames:
        # Each frame refer to a mode
        patches = generate_patch_collection(
            block_centroids=block_centroids,
            centroid_node_vectors=centroid_node_vectors,
            block_displacements=block_displacements[i, :, :] * scale_deformation,
            field_values=field_values[i],
            deformed=deformed,
            clim=None,  # Normalize colors between min and max
        )
        axes.clear()
        axes.set_title(rf"$\Omega={data.eigenvalues[i]:.4f}$", fontsize=fontsize)
        axes.add_collection(patches)
        axes.set(xlim=xlim, ylim=ylim)

        if mesh == True:
            n1 = geometry.n1_blocks
            n2 = geometry.n2_blocks
            for j in np.arange(geometry.n2_blocks):
                row_block_coordinates = np.array(
                    [
                        block_centroids[n1 * j: n1 * (j + 1), 0]
                        + block_displacements[i, n1 * j: n1 * (j + 1), 0]
                        * scale_deformation,
                        block_centroids[n1 * j: n1 * (j + 1), 1]
                        + block_displacements[i, n1 * j: n1 * (j + 1), 1]
                        * scale_deformation,
                    ]
                )
                axes.plot(row_block_coordinates[0, :], row_block_coordinates[1, :], "k")

            for k in np.arange(geometry.n1_blocks):
                col_block_coordinates = np.array(
                    [
                        block_centroids[k: n1 * (n2 - 1) + k + 1: n1, 0]
                        + block_displacements[i, k: n1 * (n2 - 1) + k + 1: n1, 0]
                        * scale_deformation,
                        block_centroids[k: n1 * (n2 - 1) + k + 1: n1, 1]
                        + block_displacements[i, k: n1 * (n2 - 1) + k + 1: n1, 1]
                        * scale_deformation,
                    ]
                )
                axes.plot(col_block_coordinates[0, :], col_block_coordinates[1, :], "k")

        out_path = Path(f"{str(out_dir)}/{i:04d}.pdf")
        # Make sure parents directories exist
        out_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(str(out_path), dpi=dpi)

    plt.close(fig)


def generate_frames(
    data: SolutionData,
    field,
    out_dir,
    field_values=None,
    deformed=False,
    frame_range=None,
    figsize=None,
    xlim=None,
    ylim=None,
    dpi=200,
    cmap=orange_blue_cmap(),
    vlim=None,
    legend_label=None,
    fontsize=14,
    ticksize=14,
    axis=True,
    grid=False,
):
    """
    docstring
    """

    _field_values, min_value, max_value, fig, axes, frames = prepare_solution_figure(
        data,
        field,
        frame_range,
        figsize,
        cmap=cmap,
        vlim=vlim,
        legend_label=legend_label,
        fontsize=fontsize,
        ticksize=ticksize,
        axis=axis,
        field_values=field_values,
    )
    block_centroids = data.block_centroids
    centroid_node_vectors = data.centroid_node_vectors
    bond_connectivity = data.bond_connectivity
    block_displacements = data.fields[:, 0, :, :]
    clim = vlim if vlim is not None else (min_value, max_value)

    for i in frames:
        # Delete old patches
        axes.clear()
        # Draw new patches
        patches = generate_patch_collection(
            block_centroids=block_centroids,
            centroid_node_vectors=centroid_node_vectors,
            block_displacements=block_displacements[i, :, :],
            field_values=_field_values[i],
            deformed=deformed,
            clim=clim,
            cmap=cmap,
        )
        axes.add_collection(patches)
        # Generate collection of bonds as lines
        collection_bonds = generate_bond_collection(
            block_centroids,
            centroid_node_vectors,
            bond_connectivity,
            block_displacements=block_displacements[i],
            deformed=deformed,
        )
        axes.add_collection(collection_bonds)

        if not axis:
            axes.axis("off")

        out_path = Path(f"{str(out_dir)}/{i:04d}.png")
        # Make sure parents directories exist
        out_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(str(out_path), dpi=dpi)

    plt.close(fig)


def generate_animation(
    data: SolutionData,
    field,
    out_filename,
    field_values=None,
    frame_range=None,
    figsize=None,
    xlim=None,
    ylim=None,
    fps=20,
    dpi=200,
    cmap=orange_blue_cmap(),
    vlim=None,
    legend_label=None,
    fontsize=14,
    ticksize=14,
    axis=True,
    grid=True,
):
    """
    docstring
    """

    _field_values, min_value, max_value, fig, axes, frames = prepare_solution_figure(
        data,
        field,
        frame_range,
        figsize,
        cmap=cmap,
        vlim=vlim,
        legend_label=legend_label,
        fontsize=fontsize,
        ticksize=ticksize,
        axis=axis,
        field_values=field_values,
    )
    clim = vlim if vlim is not None else (min_value, max_value)
    axes.grid(grid)

    out_path = Path(f"{out_filename}.mp4")
    # Make sure parents directories exist
    out_path.parent.mkdir(parents=True, exist_ok=True)

    # Generate collection of blocks as polygons
    vertices = data.centroid_node_vectors
    centroids = data.block_centroids
    DOFs = data.fields[0, 0, :, :]
    block_coords = current_coordinates(vertices, centroids, DOFs[:, -1], DOFs[:, :2])
    collection_blocks = PolyCollection(block_coords, cmap=cmap, alpha=0.95)
    collection_blocks.set_array(_field_values[0])
    collection_blocks.set_clim(*clim)
    collection_blocks.set(edgecolor="black", linewidth=0.5)
    axes.add_collection(collection_blocks)

    if data.bond_connectivity is not None:
        # Generate collection of bonds as lines
        n_blocks, n_npb, _ = block_coords.shape
        node_coords = block_coords.reshape((n_blocks * n_npb, 2))
        collection_bonds = LineCollection(
            node_coords[data.bond_connectivity], color="black", linewidth=0.5
        )
        axes.add_collection(collection_bonds)

        axes.set(xlim=xlim, ylim=ylim)

        def animate_blocks_and_bonds(i):
            # Update blocks location
            DOFs = data.fields[i, 0, :, :]
            block_coords = current_coordinates(
                vertices, centroids, DOFs[:, -1], DOFs[:, :2]
            )
            collection_blocks.set_verts(block_coords)
            collection_blocks.set_array(_field_values[i])
            # Update bonds location
            node_coords = block_coords.reshape((n_blocks * n_npb, 2))
            collection_bonds.set_segments(node_coords[data.bond_connectivity])
            axes.set(xlim=xlim, ylim=ylim)
            return collection_blocks, collection_bonds
    else:
        # Do not draw bonds
        def animate_blocks(i):
            # Update blocks location
            DOFs = data.fields[i, 0, :, :]
            block_coords = current_coordinates(
                vertices, centroids, DOFs[:, -1], DOFs[:, :2]
            )
            collection_blocks.set_verts(block_coords)
            collection_blocks.set_array(_field_values[i])
            axes.set(xlim=xlim, ylim=ylim)
            return (collection_blocks,)

    animate = (
        animate_blocks_and_bonds
        if data.bond_connectivity is not None
        else animate_blocks
    )  # type: ignore
    anim = animation.FuncAnimation(fig, animate, frames=frames, blit=True)  # type: ignore
    anim.save(str(out_path), writer="ffmpeg", fps=fps, dpi=dpi)


def generate_several_animations_for_mechanical_cloak(
    list_data,
    field,
    trace_dof: int,
    id_block_mg: int,
    id_block_cloak_area: int,
    out_filename,
    order=["mg", "cloak", "cloak"],
    field_values=None,
    frame_range=None,
    figsize=None,
    xlim=None,
    ylim=None,
    fps=20,
    dpi=200,
    cmap=orange_blue_cmap(),
    vlim=None,
    legend_label=None,
    fontsize=14,
    ticksize=14,
    axis=True,
    grid=False,
    trace_line_width=2,
    trace_colors=["blue", "red", "green"],
    marker_size=1,
    marker_linewidth=1,
):
    """
    docstring
    """
    N = len(list_data)
    # Create a figure and axes to hold the three animations
    fig, axes = plt.subplot_mosaic(
        "ABC;ABC;DDD", figsize=figsize, constrained_layout=True
    )
    for k in axes.keys():
        if k != "D":
            axes[k].axis("equal")
        axes[k].tick_params(labelsize=ticksize)

    out_path = Path(f"{out_filename}.mp4")
    out_path.parent.mkdir(
        parents=True, exist_ok=True
    )  # Make sure parents directories exis

    # Get the number of frames for each animation
    frames = range(len(list_data[0].timepoints)) if frame_range is None else frame_range

    # Generate the data usefull for each animations
    list_field_values = []
    list_v_min_v_max = []
    list_block_coords = []
    list_collection_blocks = []
    list_collection_bonds = []

    for (k, data), ax_key in zip(enumerate(list_data), axes.keys()):
        _field_values, min_value, max_value, vmin, vmax, _legend_label = (
            prepare_solution_figure_for_several_animations(
                data,
                field,
                vlim=vlim,
                legend_label=legend_label,
                fontsize=fontsize,
                ticksize=ticksize,
                axis=axis,
                field_values=field_values,
            )
        )
        list_field_values.append(_field_values)
        list_v_min_v_max.append((vmin, vmax))

        clim = vlim if vlim is not None else (min_value, max_value)

        # Generate collection of blocks as polygons
        vertices = data.centroid_node_vectors
        centroids = data.block_centroids
        DOFs = data.fields[0, 0, :, :]
        block_coords = current_coordinates(
            vertices, centroids, DOFs[:, -1], DOFs[:, :2]
        )
        list_block_coords.append(block_coords)
        collection_blocks = PolyCollection(block_coords, cmap=cmap, alpha=0.95)
        collection_blocks.set_array(_field_values[0])
        collection_blocks.set_clim(*clim)
        collection_blocks.set(edgecolor="black", linewidth=0.5)
        list_collection_blocks.append(collection_blocks)
        axes[ax_key].add_collection(collection_blocks)

        # generate the bonds
        n_blocks, n_npb, _ = list_block_coords[k].shape
        node_coords = list_block_coords[k].reshape((n_blocks * n_npb, 2))
        collection_bonds = LineCollection(
            node_coords[data.bond_connectivity], color="black", linewidth=0.5
        )
        list_collection_bonds.append(collection_bonds)
        axes[ax_key].add_collection(collection_bonds)
        axes[ax_key].set_xlim(xmin=xlim[0], xmax=xlim[1])
        axes[ax_key].set_ylim(ymin=ylim[0], ymax=ylim[1])

    labels = ["Intact", "Holed", "Cloaked"]
    plot_traces = []
    for k, data in enumerate(list_data):
        if order[k] == "mg":
            id_block = id_block_mg
        else:
            id_block = id_block_cloak_area
        (plot_trace,) = axes["D"].plot(
            data.timepoints[frames[0]:],
            data.fields[frames[0]:, 0, id_block, trace_dof],
            label=labels[k],
            linewidth=trace_line_width,
            color=trace_colors[k],
        )
        plot_traces.append(plot_trace)
    axes["D"].axhline(0, color="black", linewidth=1.0)
    axes["D"].set_ylabel(r"Displacement [mm]", fontsize=fontsize)
    # Fix position of y label
    axes["D"].yaxis.set_label_coords(-0.04, 0.5)
    axes["D"].set_ymargin(0.08)
    axes["D"].set_xlabel("Time [s]", fontsize=fontsize)
    axes["D"].tick_params(labelsize=ticksize)
    axes["D"].legend(loc="upper right", fontsize=fontsize, frameon=True, framealpha=0.8)
    for spine in axes["D"].spines.values():
        spine.set_linewidth(1.0)

    # Draw a circle at the id_block_mg and id_block_cloak_area
    for (k, data), ax_key in zip(enumerate(list_data), axes.keys()):
        if order[k] == "mg":
            id_block = id_block_mg
        else:
            id_block = id_block_cloak_area
        circle = plt.Circle(
            (data.block_centroids[id_block, 0], data.block_centroids[id_block, 1]),
            marker_size,
            fill=False,
            linewidth=marker_linewidth,
            color=trace_colors[k],
        )
        axes[ax_key].add_artist(circle)

    if not axis:
        for k in axes.keys():
            if k != "D":
                axes[k].axis("off")
    if not grid:
        for k in axes.keys():
            axes[k].grid(False)
    vmin = np.min(np.array([list_v_min_v_max[k][0] for k in range(3)]))
    vmax = np.max(np.array([list_v_min_v_max[k][1] for k in range(3)]))

    cb = fig.colorbar(
        cm.ScalarMappable(cmap=cmap, norm=colors.Normalize(vmin=vmin, vmax=vmax)),
        pad=0.02,
        label=_legend_label,
        aspect=25,
        ax=[axes["A"], axes["B"], axes["C"]],
    )
    cb.ax.tick_params(labelsize=ticksize)
    cb.ax.set_ylabel(_legend_label, fontsize=fontsize)

    def animate(i):
        for (k, data), ax_key in zip(enumerate(list_data), axes.keys()):
            # Update blocks location
            vertices = list_data[k].centroid_node_vectors
            centroids = list_data[k].block_centroids
            DOFs = list_data[k].fields[i, 0, :, :]
            block_coords = current_coordinates(
                vertices, centroids, DOFs[:, -1], DOFs[:, :2]
            )
            list_collection_blocks[k].set_verts(block_coords)
            list_collection_blocks[k].set_array(list_field_values[k][i])
            # Update bonds location
            n_blocks, n_npb, _ = list_block_coords[k].shape
            node_coords = block_coords.reshape((n_blocks * n_npb, 2))
            list_collection_bonds[k].set_segments(
                node_coords[list_data[k].bond_connectivity]
            )
            axes[ax_key].set(xlim=xlim, ylim=ylim)

        for (k, data), plot_trace in zip(enumerate(list_data), plot_traces):
            if order[k] == "mg":
                id_block = id_block_mg
            else:
                id_block = id_block_cloak_area
            plot_trace.set_data(
                data.timepoints[frames[0]: i],
                data.fields[frames[0]: i, 0, id_block, trace_dof],
            )  # , label=labels[k])
        return *list_collection_blocks, *list_collection_bonds

    # Create the combined animation using FuncAnimation
    anim = animation.FuncAnimation(fig, animate, frames=frames, blit=True)  # type: ignore

    # Display or save the combined animation
    anim.save(str(out_path), writer="ffmpeg", fps=fps, dpi=dpi)


def generate_several_animations(
    list_data,
    field,
    out_filename,
    row_or_column: str,
    field_values=None,
    frame_range=None,
    figsize=None,
    xlim=None,
    ylim=None,
    fps=20,
    dpi=200,
    cmap=orange_blue_cmap(),
    vlim=None,
    legend_label=None,
    fontsize=14,
    ticksize=14,
    axis=True,
):
    """
    docstring
    """
    N = len(list_data)
    # Create a figure and axes to hold the three animations
    if row_or_column == "row":
        fig, axes = plt.subplots(1, N, figsize=figsize, constrained_layout=True)
        def from_k_to_window_coordinate(k): return k

        def get_solution_data(list_data, window_coordinate):
            return list_data[window_coordinate]
    elif row_or_column == "column":
        fig, axes = plt.subplots(N, 1, figsize=figsize, constrained_layout=True)
        def from_k_to_window_coordinate(k): return k

        def get_solution_data(list_data, window_coordinate):
            return list_data[window_coordinate]
    elif row_or_column == "both":
        shape = (len(list_data), len(list_data[0]))
        fig, axes = plt.subplots(
            shape[0], shape[1], figsize=figsize, constrained_layout=True
        )
        N = shape[0] * shape[1]

        def from_k_to_window_coordinate(k):
            return k // shape[1], k % shape[1]

        def get_solution_data(list_data, window_coordinate):
            i, j = window_coordinate
            return list_data[i][j]

    for k in range(N):
        window_coordinate = from_k_to_window_coordinate(k)
        axes[window_coordinate].axis("equal")
        axes[window_coordinate].tick_params(labelsize=ticksize)
        out_path = Path(f"{out_filename}.mp4")
        out_path.parent.mkdir(
            parents=True, exist_ok=True
        )  # Make sure parents directories exis

    # Generate the data usefull for each animations
    list_field_values = []
    list_v_min_v_max = []
    list_block_coords = []
    list_collection_blocks = []
    list_collection_bonds = []

    for k in range(N):
        window_coordinate = from_k_to_window_coordinate(k)
        data = get_solution_data(list_data, window_coordinate)
        if field_values == None:
            _field_values, min_value, max_value, vmin, vmax, _legend_label = (
                prepare_solution_figure_for_several_animations(
                    data,
                    field,
                    vlim=vlim,
                    legend_label=legend_label,
                    fontsize=fontsize,
                    ticksize=ticksize,
                    axis=axis,
                    field_values=field_values,
                )
            )
        else:
            _field_values, min_value, max_value, vmin, vmax, _legend_label = (
                prepare_solution_figure_for_several_animations(
                    data,
                    field,
                    vlim=vlim,
                    legend_label=legend_label,
                    fontsize=fontsize,
                    ticksize=ticksize,
                    axis=axis,
                    field_values=field_values[k],
                )
            )
        list_field_values.append(_field_values)
        list_v_min_v_max.append((vmin, vmax))

        clim = vlim if vlim is not None else (min_value, max_value)

        # Generate collection of blocks as polygons
        vertices = data.centroid_node_vectors
        centroids = data.block_centroids
        DOFs = data.fields[0, 0, :, :]
        block_coords = current_coordinates(
            vertices, centroids, DOFs[:, -1], DOFs[:, :2]
        )
        list_block_coords.append(block_coords)
        collection_blocks = PolyCollection(block_coords, cmap=cmap, alpha=0.95)
        collection_blocks.set_array(_field_values[0])
        collection_blocks.set_clim(*clim)
        collection_blocks.set(edgecolor="black", linewidth=0.5)
        list_collection_blocks.append(collection_blocks)
        axes[window_coordinate].add_collection(collection_blocks)

        # Generate collection of bonds as lines
        n_blocks, n_npb, _ = list_block_coords[k].shape
        node_coords = list_block_coords[k].reshape((n_blocks * n_npb, 2))
        collection_bonds = LineCollection(
            node_coords[data.bond_connectivity], color="black", linewidth=0.5
        )
        list_collection_bonds.append(collection_bonds)
        axes[window_coordinate].add_collection(collection_bonds)
        axes[window_coordinate].set(xlim=xlim, ylim=ylim)

    if not axis:
        for k in range(N):
            window_coordinate = from_k_to_window_coordinate(k)
            axes[window_coordinate].axis("off")
    vmin = np.min(np.array([list_v_min_v_max[k][0] for k in range(N)]))
    vmax = np.max(np.array([list_v_min_v_max[k][1] for k in range(N)]))

    # fig.subplots_adjust(right=0.8)
    # cbar_ax = fig.add_axes([0.85, 0.15, 0.05, 0.7])
    cb = fig.colorbar(
        cm.ScalarMappable(cmap=cmap, norm=colors.Normalize(vmin=vmin, vmax=vmax)),
        pad=0.02,
        label=_legend_label,
        aspect=30 if row_or_column == "row" else 60,
        ax=axes,
        # cax=cbar_ax
        # location='bottom'
    )
    cb.ax.tick_params(labelsize=ticksize)
    cb.ax.set_ylabel(_legend_label, fontsize=fontsize)
    # Get the number of frames for each animation
    data0 = get_solution_data(list_data, from_k_to_window_coordinate(0))
    frames = range(len(data0.timepoints)) if frame_range is None else frame_range

    # Define the update function to be called for each frame
    # def update(frame):
    #     # Clear the axes
    #     for ax in axes:
    #         ax.cla()

    #     # Draw each animation frame on the corresponding axis
    #     axes[0].imshow(animation1[frame])
    #     axes[1].imshow(animation2[frame])
    #     axes[2].imshow(animation3[frame])

    #     # Add any necessary titles, labels, etc.

    def animate_blocks_and_bonds(i):
        for k in range(N):
            window_coordinate = from_k_to_window_coordinate(k)
            data = get_solution_data(list_data, window_coordinate)
            # Update blocks location
            vertices = data.centroid_node_vectors
            centroids = data.block_centroids
            DOFs = data.fields[i, 0, :, :]
            block_coords = current_coordinates(
                vertices, centroids, DOFs[:, -1], DOFs[:, :2]
            )
            list_collection_blocks[k].set_verts(block_coords)
            list_collection_blocks[k].set_array(list_field_values[k][i])
            # Update bonds location
            n_blocks, n_npb, _ = list_block_coords[k].shape
            node_coords = block_coords.reshape((n_blocks * n_npb, 2))
            list_collection_bonds[k].set_segments(node_coords[data.bond_connectivity])
            axes[window_coordinate].set(xlim=xlim, ylim=ylim)
        return *list_collection_blocks, *list_collection_bonds

    # Create the combined animation using FuncAnimation
    animate = animate_blocks_and_bonds  # type: ignore
    anim = animation.FuncAnimation(fig, animate, frames=frames, blit=True)  # type: ignore

    # Display or save the combined animation
    anim.save(str(out_path), writer="ffmpeg", fps=fps, dpi=dpi)


def plot_video_frame_field_overlaid(
    video_filename: Union[str, Path],
    solution_data: SolutionData,
    frame_number: int,
    timepoint: int,
    field: str,
    calib_xy: Tuple[float, float],
    ROI_X: Tuple[float, float] = (0, -1),
    ROI_Y: Tuple[float, float] = (0, -1),
    field_values: Optional[np.ndarray] = None,
    vmin: Optional[float] = None,
    vmax: Optional[float] = None,
    alpha_overlay=0.8,
    shift_px=(0, 0),
    cmap="inferno",
    figsize=(8, 5),
    ax=None,
):
    """Plot a frame of the video overlaid with the field values of the blocks.

    Args:
        video_filename (Union[str, Path]): Path to the video file.
        solution_data (SolutionData): Solution data.
        frame_number (int): Frame number to plot.
        timepoint (int): Timepoint of the solution data.
        field (str): Field to plot.
        calib_xy (Tuple[float, float]): Calibration factors for x and y.
        ROI_X (tuple[int, int], optional): ROI in the x-direction. If -1 is provided, the whole frame will be used. Defaults to (0, -1).
        ROI_Y (tuple[int, int], optional): ROI in the y-direction. If -1 is provided, the whole frame will be used. Defaults to (0, -1).
        field_values (Optional[np.ndarray], optional): Field values. Defaults to None.
        vmin (Optional[float], optional): Minimum value of the field. Defaults to None.
        vmax (Optional[float], optional): Maximum value of the field. Defaults to None.
        alpha_overlay (float, optional): Alpha of the overlay. Defaults to 0.8.
        shift_px (Tuple[int, int], optional): Shift in pixels for alignment. Defaults to (0, 0).
        cmap (str, optional): Colormap. Defaults to "inferno".
        figsize (Tuple[int, int], optional): Figure size. Defaults to (8, 5).
        ax ([type], optional): Axes. Defaults to None.

    Returns:
        Tuple[plt.Figure, plt.Axes]: Figure and axes.
    """

    # Load the video using opencv
    video = cv2.VideoCapture(f"{video_filename}")
    # Get frame number
    video.set(cv2.CAP_PROP_POS_FRAMES, frame_number)
    # Read the frame
    _, frame = video.read()
    # Add alpha channel
    frame = cv2.cvtColor(frame, cv2.COLOR_RGB2RGBA)
    # Restrict the frame to the ROI
    frame = cv2.flip(frame, 0)
    ROI_X = (ROI_X[0], ROI_X[1] if ROI_X[1] > 0 else frame.shape[1])
    ROI_Y = (ROI_Y[0], ROI_Y[1] if ROI_Y[1] > 0 else frame.shape[0])
    flipped_ROI_Y = (frame.shape[0] - ROI_Y[1], frame.shape[0] - ROI_Y[0])
    ROI_XY = (ROI_X, flipped_ROI_Y)
    frame = frame[ROI_XY[1][0]: ROI_XY[1][1], ROI_XY[0][0]: ROI_XY[0][1]]
    shift_px = np.array(shift_px)

    # Compute current configuration of the blocks
    block_coordinates = current_coordinates(
        vertices=solution_data.centroid_node_vectors,
        centroids=solution_data.block_centroids,
        angles=solution_data.fields[timepoint, 0, :, 2],
        displacements=solution_data.fields[timepoint, 0, :, :2],
    )
    # Compute the field values
    field_values_all_times = (
        compute_field_values(solution_data, field)
        if field_values is None
        else field_values
    )
    field_values_min = field_values_all_times.min() if vmin is None else vmin
    field_values_max = field_values_all_times.max() if vmax is None else vmax
    _field_values = field_values_all_times[timepoint]
    # Make a colormap
    cmap = plt.get_cmap(cmap)
    # Normalize the field values
    norm = plt.Normalize(vmin=field_values_min, vmax=field_values_max)
    # Map the normalized values to colors
    field_colors = cmap(norm(_field_values))
    # Draw the blocks
    overlay = frame.copy()
    for block, color in zip(block_coordinates, field_colors):
        # Convert the block coordinates to pixels
        block_px = (np.array(block) / calib_xy[0]).astype(int) + shift_px
        # Draw the shape with the color and opacity 0.5
        cv2.fillPoly(
            overlay,
            pts=[block_px],
            # Color the block according to the field value
            color=(color[0] * 255, color[1] * 255, color[2] * 255, 255),
        )
    # Add the overlay to the frame
    frame = cv2.addWeighted(overlay, alpha_overlay, frame, 1 - alpha_overlay, 0)

    # Show the frame
    if ax is None:
        fig, ax = plt.subplots(figsize=figsize)

    ax.imshow(frame, origin="lower")
    ax.axis("off")
    # TODO:
    # Add colorbar option
    # Add timestamp label option
    fig = ax.get_figure()

    return fig, ax
