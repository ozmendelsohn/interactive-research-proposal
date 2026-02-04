"""
Gaussian Process Interactive Visualizations
"""

import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
from numpy.linalg import inv, cholesky, det, lstsq
from scipy.optimize import minimize


def kernel(X1, X2, l=1.0, sigma_f=1.0):
    """Isotropic squared exponential (RBF) kernel."""
    sqdist = np.sum(X1**2, 1).reshape(-1, 1) + np.sum(X2**2, 1) - 2 * np.dot(X1, X2.T)
    return sigma_f**2 * np.exp(-0.5 / l**2 * sqdist)


def laplace_kernel(X1, X2, l=1.0, sigma_f=1.0):
    """Laplacian kernel."""
    dist = np.sqrt(np.sum(X1**2, 1).reshape(-1, 1) + np.sum(X2**2, 1) - 2 * np.dot(X1, X2.T) + 1e-10)
    return sigma_f**2 * np.exp(-dist / l)


def polynomial_kernel(X1, X2, d=2, c=1.0):
    """Polynomial kernel."""
    return (np.dot(X1, X2.T) + c)**d


def posterior_predictive(X_s, X_train, Y_train, l=1.0, sigma_f=1.0, sigma_y=1e-8, kernel_func=kernel):
    """Compute GP posterior predictive distribution."""
    K = kernel_func(X_train, X_train, l, sigma_f) + sigma_y**2 * np.eye(len(X_train))
    K_s = kernel_func(X_train, X_s, l, sigma_f)
    K_ss = kernel_func(X_s, X_s, l, sigma_f) + 1e-8 * np.eye(len(X_s))
    K_inv = inv(K)

    mu_s = K_s.T.dot(K_inv).dot(Y_train)
    cov_s = K_ss - K_s.T.dot(K_inv).dot(K_s)

    return mu_s, cov_s


def nll_fn(X_train, Y_train, noise, kernel_func=kernel):
    """Negative log-likelihood for hyperparameter optimization."""
    def nll(theta):
        K = kernel_func(X_train, X_train, l=theta[0], sigma_f=theta[1]) + noise**2 * np.eye(len(X_train))
        return (0.5 * np.log(det(K)) +
                0.5 * Y_train.T.dot(inv(K).dot(Y_train)) +
                0.5 * len(X_train) * np.log(2 * np.pi)).item()
    return nll


def render():
    st.header("2. Gaussian Processes")

    st.markdown("""
    Gaussian Process (GP) is a method for predicting $y^*$ for a given $x^*$.
    GP assumes that $p(f(x_1),...,f(x_N))$ is jointly Gaussian.
    """)

    tab1, tab2, tab3 = st.tabs(["2.1 GP Theory", "2.2 Kernels", "2.3 Interactive GP"])

    with tab1:
        render_gp_theory()

    with tab2:
        render_kernels()

    with tab3:
        render_interactive_gp()


def render_gp_theory():
    st.subheader("2.1 GP Formulation")

    st.markdown("The GP predictive distribution:")

    st.latex(r"""
    y^* \sim \mathcal{N}\left(\mu(x^*), \Sigma(x^*)\right)
    """)

    st.latex(r"""
    \begin{pmatrix} \mathbf{f} \\ f^* \end{pmatrix} \sim
    \mathcal{N}\left(
    \begin{pmatrix} \mathbf{\mu} \\ \mu^* \end{pmatrix},
    \begin{pmatrix} \mathbf{K} & \mathbf{k^*} \\ {\mathbf{k^*}}^T & k^{**} \end{pmatrix}
    \right)
    """)

    st.markdown("""
    **Key Components:**
    - $\\mathbf{f}$ - Vector of observed $y_i$ values
    - $f^*$ - Prediction for point $x^*$
    - $\\mathbf{K}$ - Covariance matrix of observed points
    - $\\mathbf{k^*}$ - Kernel vector $k^* = k(x^*, x_i)$
    - $k^{**}$ - Self kernel $k^{**} = k(x^*, x^*)$
    """)

    st.subheader("Conditional Distribution")

    st.latex(r"f(x^*) = {\mathbf{k^*}}^T \mathbf{K}^{-1} \mathbf{y} = \sum_{i=1}^{N} \alpha_i k(x_i, x^*)")

    st.markdown("For noisy observations with $\\mathbf{y} = \\mathbf{f} + \\boldsymbol\\epsilon$:")

    st.latex(r"f(x^*) = {\mathbf{k^*}}^T \mathbf{K_y}^{-1} \mathbf{y}")
    st.latex(r"\text{where } \mathbf{K}_y = \mathbf{K} + \sigma_y^2 \mathbf{I}")


def render_kernels():
    st.subheader("2.2 Kernel Functions")

    st.markdown("""
    The kernel function determines how strongly the value $f(x_i)$ is coupled to point $f(x_j)$.
    """)

    kernel_type = st.selectbox(
        "Select Kernel",
        ["RBF (Squared Exponential)", "Laplacian", "Polynomial"]
    )

    col1, col2 = st.columns([1, 2])

    with col1:
        if kernel_type == "RBF (Squared Exponential)":
            st.latex(r"""
            k(x_i, x_j) = \sigma_f^2 \exp\left(-\frac{\|x_i - x_j\|^2}{2l^2}\right)
            """)
            l = st.slider("Length scale (l)", 0.1, 3.0, 1.0, 0.1, key="rbf_l")
            sigma_f = st.slider("Signal variance (σf)", 0.1, 2.0, 1.0, 0.1, key="rbf_sf")
            kern_func = lambda X1, X2: kernel(X1, X2, l, sigma_f)

        elif kernel_type == "Laplacian":
            st.latex(r"""
            k(x_i, x_j) = \sigma_f^2 \exp\left(-\frac{\|x_i - x_j\|}{l}\right)
            """)
            l = st.slider("Length scale (l)", 0.1, 3.0, 1.0, 0.1, key="lap_l")
            sigma_f = st.slider("Signal variance (σf)", 0.1, 2.0, 1.0, 0.1, key="lap_sf")
            kern_func = lambda X1, X2: laplace_kernel(X1, X2, l, sigma_f)

        else:  # Polynomial
            st.latex(r"""
            k(x_i, x_j) = (x_i \cdot x_j + c)^d
            """)
            d = st.slider("Degree (d)", 1, 5, 2, 1, key="poly_d")
            c = st.slider("Constant (c)", 0.0, 2.0, 1.0, 0.1, key="poly_c")
            kern_func = lambda X1, X2: polynomial_kernel(X1, X2, d, c)

    with col2:
        # Generate sample points
        n_sample = 50
        x = np.linspace(0, 2 * np.pi, n_sample).reshape(-1, 1)

        # Compute kernel matrix
        K = kern_func(x, x)

        fig, axes = plt.subplots(1, 2, figsize=(12, 5))

        # Kernel heatmap
        im = axes[0].imshow(K, cmap='Blues', aspect='auto')
        axes[0].set_title(f'{kernel_type} Kernel Matrix')
        axes[0].set_xlabel('Sample index')
        axes[0].set_ylabel('Sample index')
        plt.colorbar(im, ax=axes[0])

        # Kernel slice
        mid = n_sample // 2
        axes[1].plot(x.ravel(), K[mid, :], 'b-', linewidth=2)
        axes[1].axvline(x=x[mid], color='r', linestyle='--', alpha=0.5)
        axes[1].set_title(f'Kernel slice at x = {x[mid, 0]:.2f}')
        axes[1].set_xlabel('x')
        axes[1].set_ylabel('k(x_mid, x)')
        axes[1].grid(True, alpha=0.3)

        plt.tight_layout()
        st.pyplot(fig)
        plt.close()


def render_interactive_gp():
    st.subheader("2.3 Interactive Gaussian Process")

    st.markdown("""
    Explore how Gaussian Process regression works by adjusting the parameters and training points.
    """)

    col1, col2 = st.columns([1, 2])

    with col1:
        st.markdown("**Function to approximate:**")
        func_type = st.selectbox(
            "Target function",
            ["exp(cos(x))", "sin(x)", "x^2", "tanh(x)"]
        )

        if func_type == "exp(cos(x))":
            f = lambda x: np.exp(np.cos(x))
        elif func_type == "sin(x)":
            f = lambda x: np.sin(x)
        elif func_type == "x^2":
            f = lambda x: x**2
        else:
            f = lambda x: np.tanh(x)

        st.markdown("**GP Parameters:**")
        noise = st.slider("Noise level", 0.01, 0.5, 0.1, 0.01, key="gp_noise")
        n_train = st.slider("Training points", 3, 20, 7, 1, key="gp_ntrain")

        st.markdown("**Training point locations:**")
        x_range = st.slider("X range", -5.0, 5.0, (-3.0, 4.0), 0.5, key="gp_range")

        optimize = st.checkbox("Optimize hyperparameters", value=True)

    with col2:
        # Generate training data
        np.random.seed(42)  # For reproducibility
        X_train = np.linspace(x_range[0], x_range[1], n_train).reshape(-1, 1)
        Y_train = f(X_train) + noise * np.random.randn(*X_train.shape)

        # Test points
        X = np.linspace(-5, 5, 200).reshape(-1, 1)

        # Optimize hyperparameters if requested
        if optimize:
            try:
                res = minimize(nll_fn(X_train, Y_train, noise),
                              np.array([1, 1]),
                              bounds=((1e-5, None), (1e-5, None)),
                              method='L-BFGS-B')
                l_opt, sigma_f_opt = res.x
            except:
                l_opt, sigma_f_opt = 1.0, 1.0
        else:
            l_opt, sigma_f_opt = 1.0, 1.0

        # Compute posterior
        mu_s, cov_s = posterior_predictive(X, X_train, Y_train,
                                           l=l_opt, sigma_f=sigma_f_opt,
                                           sigma_y=noise)

        # Plot
        fig, ax = plt.subplots(figsize=(10, 6))

        # Ground truth
        ax.plot(X, f(X), 'k:', linewidth=2, label='Ground Truth')

        # GP prediction
        mu_s = mu_s.ravel()
        uncertainty = 1.96 * np.sqrt(np.diag(cov_s))

        ax.fill_between(X.ravel(), mu_s + uncertainty, mu_s - uncertainty,
                        alpha=0.2, color='C0', label='95% confidence')
        ax.plot(X, mu_s, 'C0-', linewidth=2, label='GP Mean')

        # Training points
        ax.scatter(X_train, Y_train, c='red', s=100, zorder=5,
                   edgecolors='black', label='Training points')

        ax.set_xlabel('x', fontsize=12)
        ax.set_ylabel('y', fontsize=12)
        ax.set_title(f'Gaussian Process Regression (l={l_opt:.2f}, σf={sigma_f_opt:.2f})')
        ax.legend(loc='best')
        ax.grid(True, alpha=0.3)
        ax.set_xlim([-5, 5])

        st.pyplot(fig)
        plt.close()

        # Show hyperparameters
        if optimize:
            st.info(f"Optimized hyperparameters: length scale = {l_opt:.3f}, signal variance = {sigma_f_opt:.3f}")
