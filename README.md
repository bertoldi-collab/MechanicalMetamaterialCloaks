# Nonlinear Mechanical Metamaterial Cloaks

![Made with Python](https://img.shields.io/badge/Made%20with-Python-blue?logo=python&logoColor=ecf0f1&labelColor=34495e)
[![Paper](https://img.shields.io/badge/Paper-10.1002/adfm.202522895-blue?logoColor=ecf0f1&labelColor=34495e)](https://doi.org/10.1002/adfm.202522895)
[![arXiv](https://img.shields.io/badge/arXiv-2508.21277-b31b1b?logo=arXiv&logoColor=arXiv&labelColor=34495e)](https://doi.org/10.48550/arXiv.2508.21277)
[![Data](https://img.shields.io/badge/Data-10.5281/zenodo.16952370-blue?logo=zenodo&logoColor=ecf0f1&labelColor=34495e)](https://doi.org/10.5281/zenodo.16952370)
[![GitHub license](https://img.shields.io/github/license/bertoldi-collab/MechanicalMetamaterialCloaks?labelColor=34495e)](LICENSE)
![Badge](https://hitscounter.dev/api/hit?url=https%3A%2F%2Fgithub.com%2Fbertoldi-collab%2FMechanicalMetamaterialCloaks&label=Visits&icon=heart-fill&color=%232ecc71&message=&style=flat&tz=UTC)

**Inverse-design of nonlinear mechanical metamaterial cloaks**

https://github.com/user-attachments/assets/e95a7768-0281-4fd5-a08a-de5e76494f11


## 🌅 Why nonlinear mechanical metamaterial cloaks?

We propose a design strategy that extends mechanical cloaking to the nonlinear regime, bypassing the limitations of traditional methods.
By framing cloaking as a *behavior-mimicking* optimization problem—matching the nonlinear mechanical
response of a reference system—we eliminate the need for explicit analytical solutions or transformation-based approaches.
We solve this optimization using the differentiable simulation framework [DifFlexMM](https://github.com/bertoldi-collab/DifFlexMM), specialized here for cloaking applications.
Besides accounting for large deformations and contact interactions, our approach brings different design challenges under the same behavior-mimicking formulation, such as shielding against external excitations and creating stress-free target regions.

## 🚁 Overview

This repository contains the full pipeline used in the paper to design and
validate nonlinear mechanical cloaks with rigid units and elastic couplings.
Key building blocks include:

- 🧩 Geometry parametrizations for cloaks and inclusions
	([mechanicalmetamaterialcloaks/geometry.py](mechanicalmetamaterialcloaks/geometry.py)).
- 🎈 Elastic energy models for ligaments and interactions
	([mechanicalmetamaterialcloaks/energy.py](mechanicalmetamaterialcloaks/energy.py)).
- 💥 Differentiable dynamic solver for forward simulation
	([mechanicalmetamaterialcloaks/dynamics.py](mechanicalmetamaterialcloaks/dynamics.py)).
- 🧭 Loading, actuation, and boundary conditions
	([mechanicalmetamaterialcloaks/loading.py](mechanicalmetamaterialcloaks/loading.py)).
- 📈 Visualization utilities
	([mechanicalmetamaterialcloaks/plotting.py](mechanicalmetamaterialcloaks/plotting.py)).

Problem definitions live in [problems](problems) and are paired with notebooks
that reproduce each cloaking task and experimental dataset.

## 📜 Paper

This repository contains all the code developed for the paper:

> [G. Bordiga, J.-G. Argaud, A. A. Watkins, V. Tournat, K. Bertoldi.
> Nonlinear Mechanical Metamaterial Cloaks.
> _Advanced Functional Materials_ (2025).](https://doi.org/10.1002/adfm.202522895)

Preprint: https://doi.org/10.48550/arXiv.2508.21277

## 🎯 Solved design problems

|  | Task description | Notebooks | Data 💾 | Experiments 🧪 | Video 🎥 |
| --- | --- | --- | --- | --- | --- |
| 🛡️ | Blocking unwanted excitations (dynamic shielding) | [Shielding](notebooks/quads_dynamic_cloaking_3dp_pla_shim_shelding_source_multi_loadings.ipynb)<br>[Cloak size sweep](notebooks/quads_dynamic_cloaking_3dp_pla_shim_shelding_source_multi_loadings_cloak_size.ipynb) | [Cloaked](data/quads_shielding_multi_loading_source_3dp_pla_shims_4_loads)<br>[Reference](data/quads_shielding_multi_loading_source_3dp_pla_shims_4_loads_reference) | [Cloaked](exp/quads_shielding_multi_loading_source_3dp_pla_shims_4_loads)<br>[Reference](exp/quads_shielding_multi_loading_source_3dp_pla_shims_4_loads_reference) | [Video](https://github.com/user-attachments/assets/6d60d8b1-92e4-48a5-867a-e0f2c5bfefb0) |
| 🫥 | Stress-free regions | [Stress-free circle](notebooks/quads_static_cloaking_3dp_pla_shims_circle_stress_free.ipynb) | [Stress-free](data/quads_static_cloaking_3dp_pla_shims_stress_free) | [Stress-free](exp/quads_static_cloaking_3dp_pla_shims_stress_free) | [Video](https://github.com/user-attachments/assets/d71ae8da-c484-4ca0-ab6b-ba650550cc1b) |
| 🕳️ | Static cloaking of voids | [Heart](notebooks/quads_static_cloaking_3dp_pla_shims_heart_big_domain.ipynb)<br>[Cat](notebooks/quads_static_cloaking_3dp_pla_shims_cat_big_domain.ipynb)<br>[Dolphin](notebooks/quads_static_cloaking_3dp_pla_shims_dolphin_big_domain.ipynb)<br>[Shamrock](notebooks/quads_static_cloaking_3dp_pla_shims_shamrock_big_domain.ipynb)<br>[Smile](notebooks/quads_static_cloaking_3dp_pla_shims_big_smile.ipynb)<br>[Kagome](notebooks/kagome_static_cloaking_3dp_pla_shims.ipynb) | [Quads](data/quads_static_cloaking_3dp_pla_shims)<br>[Reference](data/quads_static_cloaking_3dp_pla_shims_reference_compressed)<br>[Kagome](data/kagome_static_cloaking_3dp_pla_shims) | [Reference](exp/quads_static_cloaking_3dp_pla_shims_reference_compressed)<br>[Smile void](exp/quads_static_cloaking_3dp_pla_shims_void) | [Video](https://github.com/user-attachments/assets/dd133650-8263-4370-85b0-4e6273caee3a) |
| 🧱 | Static cloaking of rigid inclusions | [Rigid inclusion](notebooks/quads_static_cloaking_3dp_pla_shims_heart_rigid_inclusion.ipynb) | [Rigid inclusion](data/quads_static_cloaking_3dp_pla_shims_rigid_inclusion) |  |  |
| 🌊 | Dynamic cloaking of voids | [Quads](notebooks/quads_dynamic_cloaking_3dp_pla_shims.ipynb) | [Quads](data/quads_dynamic_cloaking_3dp_pla_shims) |  | [Video](https://github.com/user-attachments/assets/390cf0c2-dc9c-4dff-9f09-db4bfdf4c14f) |

## 💾 Optimization and experimental data

> [!IMPORTANT]
> Install the [package along with the examples](#mechanicalmetamaterialcloaks-with-examples) to visualize the data and notebooks.


All data generated or used for the paper can be downloaded from
[![DOI](https://img.shields.io/badge/Data-10.5281/zenodo.16952370-blue?logo=zenodo&logoColor=ecf0f1&labelColor=34495e)](https://doi.org/10.5281/zenodo.16952370).
To access and visualize the data:

- Download the archive from Zenodo.
- Extract the contents into the root directory of the repository.
- Use the [notebooks](notebooks) to load and visualize optimization results.
- Explore experimental results under [exp](exp).

## ⬇️ Installation

### MechanicalMetamaterialCloaks only

Assuming you have access to the repo and ssh keys are set up in your GitHub
account, you can install the package with

```bash
pip install git+ssh://git@github.com/bertoldi-collab/MechanicalMetamaterialCloaks.git
```

### MechanicalMetamaterialCloaks with examples

Clone the repository, `cd` into the root folder, and install with

```bash
pip install -e .
```

The code has been tested with Python 3.10–3.11 but may work with other versions once the proper dependencies are installed.

## 🤝 Contributing

<details>
<summary><b>Expand here</b></summary>

The dependency management of the project is done via
[poetry](https://python-poetry.org/docs/).

To get started:

- Install [poetry](https://python-poetry.org/docs/).
- Clone the repository.
- `cd` into the root directory and run `poetry install`. This will create the
	poetry environment with all the necessary dependencies.
- If you are using VS Code, search for `venv path` in the settings and paste
	`~/.cache/pypoetry/virtualenvs` in the `venv path` field. Then select the
	poetry environment as the Python environment for the project.

</details>

## 📝 Citation

If you use this code in your research or anywhere, please cite the paper:

```bibtex
@article{bordiga2025nonlinear,
		title   = {Nonlinear Mechanical Metamaterial Cloaks},
		author  = {Bordiga, Giovanni and Argaud, Jean-Gabriel and Watkins, Audrey A. and Tournat, Vincent and Bertoldi, Katia},
		year    = {2025},
		journal = {Advanced Functional Materials},
		volume  = {36},
		number  = {28},
		pages   = {e22895},
		doi     = {10.1002/adfm.202522895},
}
```
