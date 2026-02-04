"""
Normal/Gaussian Distribution Interactive Visualizations
"""

import streamlit as st
import numpy as np
import matplotlib.pyplot as plt


def univariate_normal(x, mean, variance):
    """PDF of the univariate normal distribution."""
    return ((1. / np.sqrt(2 * np.pi * variance)) *
            np.exp(-((x - mean)**2) / (2 * variance)))


def laplace(x, mean, variance):
    return ((1. / (2 * variance)) *
            np.exp(-np.abs(x - mean) / variance))


def fermi(x, mean, variance):
    return 1 / (np.exp((x - mean) / variance) + 1)


def boltzmann(x, mean, variance):
    return 0.797 * ((x**2) * np.exp((-x**2) / (2 * variance**2))) / variance**3


def multivariate_normal_pdf(x, d, mean, covariance):
    """PDF of the multivariate normal distribution."""
    x_m = x - mean
    return (1. / (np.sqrt((2 * np.pi)**d * np.linalg.det(covariance))) *
            np.exp(-(np.linalg.solve(covariance, x_m).T.dot(x_m)) / 2))


def render():
    st.header("1. Normal/Gaussian Distribution")

    st.markdown("""
    Normal distributions are important in statistics and are often used to represent real-valued
    random variables whose distributions are not known. Their importance is partly due to the
    [central limit theorem](https://en.wikipedia.org/wiki/Central_limit_theorem).
    """)

    # Tabs for different sections
    tab1, tab2, tab3 = st.tabs(["1.1 Univariate", "1.2 Multivariate", "1.3 Conditional"])

    with tab1:
        render_univariate()

    with tab2:
        render_multivariate()

    with tab3:
        render_conditional()


def render_univariate():
    st.subheader("1.1 Univariate Normal Distribution")

    st.latex(r"\mathcal{N}(\mu, \sigma^2)")
    st.latex(r"p(x \mid \mu, \sigma) = \frac{1}{\sqrt{2\pi\sigma^2}} \exp{ \left( -\frac{(x - \mu)^2}{2\sigma^2}\right)}")

    col1, col2 = st.columns([1, 2])

    with col1:
        st.markdown("**Parameters:**")
        mu = st.slider("Mean (μ)", -3.0, 3.0, 0.0, 0.1, key="uni_mu")
        sigma = st.slider("Std Dev (σ)", 0.1, 2.0, 1.0, 0.01, key="uni_sigma")

        dist_type = st.selectbox(
            "Distribution Type",
            ["Normal", "Laplace", "Fermi", "Boltzmann"]
        )

    with col2:
        fig, ax = plt.subplots(figsize=(8, 5))
        x = np.linspace(-5, 5, 500)

        if dist_type == "Normal":
            y = univariate_normal(x, mu, sigma**2)
            label = f'$\\mathcal{{N}}({mu}, {sigma**2:.2f})$'
        elif dist_type == "Laplace":
            y = laplace(x, mu, sigma)
            label = f'Laplace({mu}, {sigma})'
        elif dist_type == "Fermi":
            y = fermi(x, mu, sigma)
            label = f'Fermi({mu}, {sigma})'
        else:  # Boltzmann
            x = np.linspace(0.01, 5, 500)
            y = boltzmann(x, mu, sigma)
            label = f'Boltzmann({mu}, {sigma})'

        ax.plot(x, y, label=label, color='C0', linewidth=2)
        ax.set_xlabel('x', fontsize=12)
        ax.set_ylabel('density: p(x)', fontsize=12)
        ax.set_title(f'{dist_type} Distribution')
        ax.legend(loc='upper right')
        ax.set_ylim([0, max(y) * 1.1])
        ax.grid(True, alpha=0.3)

        st.pyplot(fig)
        plt.close()


def render_multivariate():
    st.subheader("1.2 Multivariate Normal Distribution")

    st.latex(r"p(\mathbf{x} \mid \mathbf{\mu}, \Sigma) = \frac{1}{\sqrt{(2\pi)^d |\Sigma|}} \exp{ \left( -\frac{1}{2}(\mathbf{x} - \mathbf{\mu})^T \Sigma^{-1} (\mathbf{x} - \mathbf{\mu}) \right)}")

    st.markdown("""
    The covariance matrix determines the shape and orientation of the distribution:
    """)

    st.latex(r"""
    \mathcal{N}\left(
    \begin{bmatrix} 0 \\ 1 \end{bmatrix},
    \begin{bmatrix} 1 & C \\ C & 1 \end{bmatrix}\right)
    """)

    col1, col2 = st.columns([1, 2])

    with col1:
        C = st.slider("Correlation (C)", 0.0, 0.99, 0.8, 0.01, key="multi_c")
        nb_of_x = st.slider("Grid Resolution", 20, 100, 50, 10, key="multi_res")

    with col2:
        mean = np.array([[0.], [1.]])
        covariance = np.array([[1., C], [C, 1.]])

        x1s = np.linspace(-3, 3, nb_of_x)
        x2s = np.linspace(-1.5, 3.5, nb_of_x)
        x1, x2 = np.meshgrid(x1s, x2s)

        pdf = np.zeros((nb_of_x, nb_of_x))
        for i in range(nb_of_x):
            for j in range(nb_of_x):
                point = np.array([[x1[i, j]], [x2[i, j]]])
                pdf[i, j] = multivariate_normal_pdf(point, 2, mean, covariance)

        fig, ax = plt.subplots(figsize=(8, 6))
        contour = ax.contourf(x1, x2, pdf, levels=30, cmap='YlGnBu')
        ax.set_xlabel('$y_1$', fontsize=12)
        ax.set_ylabel('$y_2$', fontsize=12)
        ax.set_title(f'Bivariate Normal Distribution (C={C})')
        plt.colorbar(contour, ax=ax, label='$p(y_1, y_2)$')

        st.pyplot(fig)
        plt.close()


def render_conditional():
    st.subheader("1.3 Conditional Normal Distributions")

    st.markdown("""
    If $\\mathbf{y_1}$ and $\\mathbf{y_2}$ are jointly normal, the conditional distribution
    of $\\mathbf{y_1}$ given $\\mathbf{y_2}$ is:
    """)

    st.latex(r"p(\mathbf{y_1} \mid \mathbf{y_2}) = \mathcal{N}(\mu_{1|2}, \Sigma_{1|2})")
    st.latex(r"\mu_{1|2} = \mu_1 + \Sigma_{12}\Sigma_{22}^{-1}(x_2-\mu_2)")
    st.latex(r"\Sigma_{1|2} = \Sigma_{11} - \Sigma_{12}\Sigma_{22}^{-1}\Sigma_{21}")

    col1, col2 = st.columns([1, 2])

    with col1:
        y1_cond = st.slider("Condition on y1", -1.0, 1.0, -0.5, 0.1, key="cond_y1")
        y2_cond = st.slider("Condition on y2", -1.0, 2.0, 1.0, 0.1, key="cond_y2")
        C = st.slider("Correlation (C)", 0.0, 0.99, 0.8, 0.01, key="cond_c")

    with col2:
        d = 2
        mean = np.array([[0.], [1.]])
        cov = np.array([[1, C], [C, 1]])

        mean_x, mean_y = mean[0, 0], mean[1, 0]
        A, B, Cov = cov[0, 0], cov[1, 1], cov[0, 1]

        # y1 | y2
        mean_xgiveny = mean_x + (Cov * (1 / B) * (y2_cond - mean_y))
        cov_xgiveny = A - Cov * (1 / B) * Cov

        # y2 | y1
        mean_ygivenx = mean_y + (Cov * (1 / A) * (y1_cond - mean_x))
        cov_ygivenx = B - (Cov * (1 / A) * Cov)

        # Generate joint distribution
        nb_of_x = 50
        x1s = np.linspace(-3, 3, nb_of_x)
        x2s = np.linspace(-1.5, 3.5, nb_of_x)
        x1, x2 = np.meshgrid(x1s, x2s)

        pdf = np.zeros((nb_of_x, nb_of_x))
        for i in range(nb_of_x):
            for j in range(nb_of_x):
                point = np.array([[x1[i, j]], [x2[i, j]]])
                pdf[i, j] = multivariate_normal_pdf(point, 2, mean, cov)

        fig = plt.figure(figsize=(10, 8))

        # Main contour plot
        ax1 = fig.add_subplot(2, 2, 1)
        contour = ax1.contourf(x1, x2, pdf, levels=30, cmap='YlGnBu')
        ax1.axhline(y=y2_cond, color='r', linestyle='--', label=f'$y_2={y2_cond}$')
        ax1.axvline(x=y1_cond, color='b', linestyle='--', label=f'$y_1={y1_cond}$')
        ax1.set_xlabel('$y_1$')
        ax1.set_ylabel('$y_2$')
        ax1.legend()
        ax1.set_title('Joint Distribution')

        # p(y2 | y1)
        ax2 = fig.add_subplot(2, 2, 2)
        yx = np.linspace(-2, 4, 100)
        pyx = univariate_normal(yx, mean_ygivenx, cov_ygivenx)
        ax2.plot(pyx, yx, 'b-', linewidth=2)
        ax2.fill_betweenx(yx, 0, pyx, alpha=0.3)
        ax2.set_xlabel('density')
        ax2.set_ylabel('$y_2$')
        ax2.set_title(f'$p(y_2|y_1={y1_cond})$')
        ax2.set_ylim(-1.5, 3.5)

        # p(y1 | y2)
        ax3 = fig.add_subplot(2, 2, 3)
        xy = np.linspace(-3, 3, 100)
        pxy = univariate_normal(xy, mean_xgiveny, cov_xgiveny)
        ax3.plot(xy, pxy, 'r-', linewidth=2)
        ax3.fill_between(xy, 0, pxy, alpha=0.3, color='red')
        ax3.set_xlabel('$y_1$')
        ax3.set_ylabel('density')
        ax3.set_title(f'$p(y_1|y_2={y2_cond})$')
        ax3.set_xlim(-3, 3)

        # Info box
        ax4 = fig.add_subplot(2, 2, 4)
        ax4.axis('off')
        info_text = f"""
        Conditional Statistics:

        $p(y_1|y_2={y2_cond})$:
          Mean = {mean_xgiveny:.3f}
          Variance = {cov_xgiveny:.3f}

        $p(y_2|y_1={y1_cond})$:
          Mean = {mean_ygivenx:.3f}
          Variance = {cov_ygivenx:.3f}
        """
        ax4.text(0.1, 0.5, info_text, fontsize=12, verticalalignment='center',
                 fontfamily='monospace', bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

        plt.tight_layout()
        st.pyplot(fig)
        plt.close()
