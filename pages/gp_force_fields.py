"""
GP Force Fields Interactive Visualizations
"""

import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, Matern, ConstantKernel


def render():
    st.header("3. GP Force Field")

    st.markdown("""
    A straightforward formulation of a vector-valued estimator for molecular forces:
    """)

    st.latex(r"""
    \mathbf{\hat{f}} = \begin{bmatrix}\hat{f}_1(x), \dots, \hat{f}_N(x)\end{bmatrix}^{T}
    """)

    st.markdown("""
    For predicting forces on $N$ atoms with 3D position vectors:
    """)

    st.latex(r"\mathbf{\hat{f}}: \mathbb{R}^{3N} \rightarrow \mathbb{R}^{3N}")

    tab1, tab2, tab3 = st.tabs(["3.1 The Problem", "3.2 Solutions", "3.3 Demo"])

    with tab1:
        render_problem()

    with tab2:
        render_solutions()

    with tab3:
        render_demo()


def render_problem():
    st.subheader("3.1 The Vector-Valued Estimator Problem")

    st.markdown("""
    When mapping a scalar function $\\mathbb{R}^{N} \\rightarrow \\mathbb{R}$, each entry in the kernel
    function corresponds to one input vector $(x_i)$. In the vector output case, we have multiple
    outputs for each input, requiring careful kernel construction.

    ### The Naive Approach

    We can create an independent estimator for each value in the force vector:
    """)

    st.latex(r"""
    \mathbf{\hat{f}} = \begin{bmatrix}\hat{f}_1(x) \\ \dots \\ \hat{f}_N(x)\end{bmatrix}
    = \begin{bmatrix}
    \mathcal{GP}[\mu(x)_1, k(x,x')_1] \\
    \dots \\
    \mathcal{GP}[\mu(x)_N, k(x,x')_N]
    \end{bmatrix}
    """)

    st.warning("""
    **Problem:** This approach treats each force component independently,
    ignoring correlations between different components and atoms.
    """)


def render_solutions():
    st.subheader("3.2 Improving the GP Force Field")

    st.markdown("### Kernel Selection")

    col1, col2 = st.columns(2)

    with col1:
        st.markdown("""
        **RBF Kernel:**
        - Infinitely differentiable
        - Very smooth predictions
        - May oversmooth for force fields
        """)

    with col2:
        st.markdown("""
        **Matern Kernel:**
        - Restricted differentiability
        - Better for physical systems
        - Parameter $\\nu$ controls smoothness
        """)

    st.latex(r"""
    k_{\nu=n+\frac{1}{2}}(d) = \exp\left(-\frac{\sqrt{2\nu}d}{\sigma}\right) P_n(d)
    """)

    st.markdown("### Output Normalization")

    st.latex(r"""
    \hat{y}_{train} = \frac{y_{train} - E[y_{train}]}{\sqrt{Var[y_{train}]}}
    """)

    st.success("""
    Normalizing output values centers them around zero with unit variance,
    reducing overexpression of certain axes.
    """)

    st.markdown("### Roto-translational Invariance")

    st.markdown("""
    Using Cartesian coordinates adds rotation and translation bias. Using distance-based
    representations removes this bias:
    """)

    st.latex(r"""
    D_{ij} = \begin{cases}
    \|r_i - r_j\| & i > j \\
    0 & i \leq j
    \end{cases}
    """)


def render_demo():
    st.subheader("3.3 GP Force Field Demo")

    st.markdown("""
    This demo shows how different GP configurations affect force prediction accuracy
    using synthetic molecular data.
    """)

    col1, col2 = st.columns([1, 2])

    with col1:
        st.markdown("**Configuration:**")

        n_atoms = st.slider("Number of atoms", 2, 10, 4, key="ff_atoms")
        n_train = st.slider("Training samples", 10, 100, 30, key="ff_train")
        n_test = st.slider("Test samples", 10, 50, 20, key="ff_test")

        kernel_type = st.selectbox(
            "Kernel",
            ["RBF", "Matern (nu=1.5)", "Matern (nu=2.5)"],
            key="ff_kernel"
        )

        normalize = st.checkbox("Normalize outputs", value=True, key="ff_norm")
        noise = st.slider("Noise level", 0.01, 0.3, 0.1, 0.01, key="ff_noise")

    with col2:
        # Generate synthetic data
        np.random.seed(42)

        # Input: flattened positions (3N dimensions)
        dim = 3 * n_atoms
        X_train = np.random.randn(n_train, dim)
        X_test = np.random.randn(n_test, dim)

        # Output: forces (3N dimensions) - synthetic harmonic forces
        def compute_forces(X):
            """Compute synthetic forces based on distances from center."""
            forces = np.zeros_like(X)
            for i in range(len(X)):
                pos = X[i].reshape(n_atoms, 3)
                center = pos.mean(axis=0)
                # Simple harmonic force towards center
                forces[i] = -(pos - center).ravel() + 0.1 * np.random.randn(dim)
            return forces

        Y_train = compute_forces(X_train)
        Y_test = compute_forces(X_test)

        # Select kernel
        if kernel_type == "RBF":
            kern = ConstantKernel(1.0) * RBF(length_scale=1.0)
        elif kernel_type == "Matern (nu=1.5)":
            kern = ConstantKernel(1.0) * Matern(length_scale=1.0, nu=1.5)
        else:
            kern = ConstantKernel(1.0) * Matern(length_scale=1.0, nu=2.5)

        # Train GP
        gpr = GaussianProcessRegressor(kernel=kern, alpha=noise**2, normalize_y=normalize)
        gpr.fit(X_train, Y_train)

        # Predict
        Y_pred_train = gpr.predict(X_train)
        Y_pred_test = gpr.predict(X_test)

        # Compute metrics
        train_norm = np.linalg.norm(Y_train, axis=1)
        test_norm = np.linalg.norm(Y_test, axis=1)
        pred_train_norm = np.linalg.norm(Y_pred_train, axis=1)
        pred_test_norm = np.linalg.norm(Y_pred_test, axis=1)

        train_error = np.mean(np.abs(train_norm - pred_train_norm) / train_norm) * 100
        test_error = np.mean(np.abs(test_norm - pred_test_norm) / test_norm) * 100

        # Plotting
        fig, axes = plt.subplots(2, 2, figsize=(12, 10))

        # Force norm comparison - Training
        axes[0, 0].scatter(range(n_train), train_norm, alpha=0.6, label='Ground Truth', c='C0')
        axes[0, 0].scatter(range(n_train), pred_train_norm, alpha=0.6, label='Predicted', c='C1')
        axes[0, 0].set_xlabel('Sample')
        axes[0, 0].set_ylabel('Force Norm')
        axes[0, 0].set_title(f'Training Set (Error: {train_error:.1f}%)')
        axes[0, 0].legend()
        axes[0, 0].grid(True, alpha=0.3)

        # Force norm comparison - Test
        axes[0, 1].scatter(range(n_test), test_norm, alpha=0.6, label='Ground Truth', c='C0')
        axes[0, 1].scatter(range(n_test), pred_test_norm, alpha=0.6, label='Predicted', c='C1')
        axes[0, 1].set_xlabel('Sample')
        axes[0, 1].set_ylabel('Force Norm')
        axes[0, 1].set_title(f'Test Set (Error: {test_error:.1f}%)')
        axes[0, 1].legend()
        axes[0, 1].grid(True, alpha=0.3)

        # Parity plot - Training
        max_val = max(train_norm.max(), pred_train_norm.max())
        axes[1, 0].scatter(train_norm, pred_train_norm, alpha=0.6, c='C2')
        axes[1, 0].plot([0, max_val], [0, max_val], 'k--', label='Perfect prediction')
        axes[1, 0].set_xlabel('True Force Norm')
        axes[1, 0].set_ylabel('Predicted Force Norm')
        axes[1, 0].set_title('Training Parity Plot')
        axes[1, 0].legend()
        axes[1, 0].grid(True, alpha=0.3)

        # Parity plot - Test
        max_val = max(test_norm.max(), pred_test_norm.max())
        axes[1, 1].scatter(test_norm, pred_test_norm, alpha=0.6, c='C3')
        axes[1, 1].plot([0, max_val], [0, max_val], 'k--', label='Perfect prediction')
        axes[1, 1].set_xlabel('True Force Norm')
        axes[1, 1].set_ylabel('Predicted Force Norm')
        axes[1, 1].set_title('Test Parity Plot')
        axes[1, 1].legend()
        axes[1, 1].grid(True, alpha=0.3)

        plt.tight_layout()
        st.pyplot(fig)
        plt.close()

        # Summary metrics
        col_a, col_b = st.columns(2)
        with col_a:
            st.metric("Training Error", f"{train_error:.2f}%")
        with col_b:
            st.metric("Test Error", f"{test_error:.2f}%")
