# CLAUDE.md - AI Assistant Guide

This document provides guidance for AI assistants working with this codebase.

## Project Overview

**Title**: Data-Driven Force Fields for Large Scale Molecular Dynamics Simulations of Halide Perovskites

**Author**: Oz Yosef Mendelsohn (ozyosef.mendelsohn@weizmann.ac.il)

**Purpose**: An interactive research proposal demonstrating the application of Gaussian Process (GP) and Gradient-Domain Machine Learning (GDML) methods to predict molecular forces in halide perovskite simulations. The project is designed to run as a Jupyter Binder environment for reproducible, interactive exploration.

## Repository Structure

```
interactive-research-proposal/
├── main_gm.ipynb          # Main interactive notebook (primary artifact)
├── ase_utils.py           # ASE molecular dynamics utilities (398 lines)
├── data_utils.py          # Data format conversion utilities (101 lines)
├── utils.py               # Gaussian Process ML utilities (452 lines)
├── requirements.txt       # Python dependencies
├── postBuild              # Binder environment setup script
├── Images/                # Documentation images
│   └── MD.png
└── *.npz                  # Pre-trained model/dataset files
```

## Module Reference

### `ase_utils.py`
ASE (Atomic Simulation Environment) integration for molecular dynamics:

| Function/Class | Purpose |
|---------------|---------|
| `GPRCalculator` | ASE Calculator using Gaussian Process regression for force prediction |
| `iGPRCalculator` | Invariant GPR Calculator using distance-based representations |
| `movingaverage()` | Exponential moving average for visualization |
| `plot_gpr()`, `plot_gdml()` | Visualization utilities for model predictions |
| `printenergy()` | Progress callback for MD simulations |
| `md_dataset_split()` | Split trajectory data into train/test sets |
| `dist()`, `dist_mat()`, `x_to_d()` | Distance matrix calculations |
| `pbc_d()` | Periodic boundary conditions handling |
| `J_D()` | Jacobian matrix for descriptor transformations |
| `create()`, `train()`, `all_script()` | SGDML command-line wrappers |

### `data_utils.py`
Converts ASE trajectory files to SGDML dataset format:

| Function | Purpose |
|----------|---------|
| `from_traj()` | Transform ASE trajectory (.traj) to SGDML dataset (.npz) |

### `utils.py`
Gaussian Process machine learning fundamentals:

| Function/Class | Purpose |
|---------------|---------|
| `univariate_plot()` | Interactive univariate normal distribution plot |
| `multivariate_normal()` | PDF of multivariate normal distribution |
| `multivariate_plot()` | Interactive bivariate normal visualization |
| `condition_plot()` | Conditional distribution visualization |
| `kernel()` | RBF (squared exponential) kernel function |
| `posterior_predictive()` | GP posterior mean and covariance computation |
| `nll_fn()` | Negative log-likelihood for hyperparameter optimization |
| `gaussian_process()` | Full GP regression workflow |
| `gaussian_process_interactive()` | Click-to-add-points GP visualization |
| `gprArray` | Wrapper for multiple independent GP regressors |
| `plot_gp()`, `plot_gp_2D()` | GP visualization helpers |

## Technology Stack

| Category | Tools |
|----------|-------|
| **Core ML** | scikit-learn (GaussianProcessRegressor), SGDML |
| **Molecular Simulation** | ASE (Atomic Simulation Environment) |
| **Numerics** | numpy, scipy |
| **Visualization** | matplotlib, nglview (3D molecular viewer) |
| **Interactive** | Jupyter, ipywidgets, ipympl |
| **Utilities** | tqdm (progress bars) |

## Development Workflow

### Running the Project

1. **Binder (recommended)**: Click the Binder badge in README.md
2. **Local setup**:
   ```bash
   pip install -r requirements.txt
   jupyter notebook main_gm.ipynb
   ```

### Code Modification Pattern

The project follows a **Jupyter-first development** pattern:
- Main narrative and experiments live in `main_gm.ipynb`
- Reusable utilities are extracted to `.py` modules
- Use `%load_ext autoreload` and `%autoreload 2` for hot-reloading during development

### Data Formats

| Format | Description |
|--------|-------------|
| `.traj` | ASE trajectory files (atomic positions, forces, energies over time) |
| `.npz` | Compressed numpy archives for SGDML datasets/models |
| `.xyz` | Extended XYZ format for molecular structures |

## Code Conventions

### Style Guidelines

- **Functions**: snake_case (`plot_gpr`, `movingaverage`, `x_to_d`)
- **Classes**: PascalCase (`GPRCalculator`, `iGPRCalculator`, `gprArray` is an exception)
- **Variables**: snake_case with clear intent
- **Docstrings**: NumPy-style when present (see `pbc_d()`, `dist()` for examples)
- **Type hints**: Not used; pure Python compatible style

### Common Patterns

1. **Calculator Pattern**: Custom ASE calculators inherit from `ase.calculators.calculator.Calculator` and implement `calculate()` method
2. **Factory Functions**: Functions like `printenergy()` return closures for use as callbacks
3. **Distance Representations**: Use `x_to_d()` to convert Cartesian coordinates to rotation-translation invariant distance descriptors

### Important Notes

- No formal test framework - validation is done through notebook experiments
- Minimal error handling - code assumes valid inputs (research-grade)
- Heavy use of numpy for all numerical computations
- Visualization uses both static matplotlib and interactive ipywidgets

## Common AI Assistant Tasks

### Adding a New Kernel Function

Add to `utils.py`:
```python
def new_kernel(X1, X2, param1=1.0, param2=1.0):
    '''
    Docstring describing the kernel.

    Args:
        X1: Array of m points (m x d).
        X2: Array of n points (n x d).

    Returns:
        Covariance matrix (m x n).
    '''
    # Implementation
    return K
```

### Creating a New ASE Calculator

Add to `ase_utils.py`:
```python
class NewCalculator(Calculator):
    implemented_properties = ['forces']  # Add 'energy' if computing energies

    def __init__(self, model, *args, **kwargs):
        super(NewCalculator, self).__init__(*args, **kwargs)
        self.model = model

    def calculate(self, atoms=None, *args, **kwargs):
        super(NewCalculator, self).calculate(atoms, *args, **kwargs)
        r = np.array(atoms.get_positions())
        # Compute forces using self.model
        f = self.model.predict(r.reshape([1, -1]))
        self.results = {'forces': f.reshape(-1, 3)}
```

### Converting New Trajectory Data

```python
from data_utils import from_traj
dataset = from_traj('path/to/trajectory.traj',
                    overwrite=True,
                    theory='DFT',
                    r_unit='Ang',
                    e_unit='eV')
```

## File Dependencies

```
main_gm.ipynb
├── utils.py          (GP fundamentals, visualizations)
├── ase_utils.py      (MD simulations, SGDML integration)
│   └── sgdml.predict.GDMLPredict
├── data_utils.py     (data conversion)
│   └── sgdml.utils.io, sgdml.utils.ui
└── *.npz             (pre-computed datasets/models)
```

## Troubleshooting

| Issue | Solution |
|-------|----------|
| `nglview` not rendering | Run: `jupyter-nbextension enable nglview --py --sys-prefix` |
| Widget not displaying | Ensure `ipywidgets` is installed and Jupyter extensions are enabled |
| SGDML import errors | Install via: `pip install sgdml` |
| ASE calculator issues | Check that forces array shape matches `(-1, 3)` |

## Git Conventions

- Commit messages are typically brief: "daily update", "Update main_gm.ipynb"
- No formal branching strategy
- Main work happens on the `main` branch
- Data files (`.npz`) may be large - consider `.gitignore` for generated files

## Key Mathematical Concepts

Understanding these concepts helps when modifying the code:

1. **Gaussian Process Regression**: Non-parametric Bayesian regression using kernel functions
2. **GDML (Gradient-Domain Machine Learning)**: Learning force fields in the gradient domain for energy conservation
3. **Periodic Boundary Conditions**: `pbc_d()` handles atoms in periodic simulation cells
4. **Jacobian Transformation**: `J_D()` transforms Cartesian gradients to distance-based descriptors
