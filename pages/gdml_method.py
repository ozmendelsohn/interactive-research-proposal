"""
Gradient-Domain Machine Learning (GDML) Method
"""

import streamlit as st
import numpy as np
import matplotlib.pyplot as plt


def render():
    st.header("4. Gradient-Domain Machine Learning (GDML)")

    st.markdown("""
    Using physical insights to build better force field kernels through the GDML method.
    """)

    tab1, tab2, tab3, tab4 = st.tabs([
        "4.1 Linear Operators",
        "4.2 Conservative Fields",
        "4.3 GDML Formulation",
        "4.4 Block Kernel"
    ])

    with tab1:
        render_linear_operators()

    with tab2:
        render_conservative_fields()

    with tab3:
        render_gdml_formulation()

    with tab4:
        render_block_kernel()


def render_linear_operators():
    st.subheader("4.1 Applying Linear Operators")

    st.markdown("""
    One way to construct the block kernel is by starting with a single-value estimator $\\hat{f}$
    and then applying a linear operator $\\hat{G}$ to obtain a vector estimator:
    """)

    st.latex(r"""
    \mathbf{\hat{g}(x)} \sim \mathcal{GP}\left[\hat{G}\mu(x), \hat{G}k(x,x')\hat{G}'^T\right]
    """)

    st.markdown("Using properties of linear operators on Gaussian distributions:")

    st.latex(r"Cov[Ax, By] = A \cdot Cov[x,y] \cdot B^T")
    st.latex(r"E[Ax] = A \cdot E[x]")

    st.info("""
    **Key Insight:** By choosing an appropriate linear operator $\\hat{G}$, we can
    transform a scalar GP into a vector-valued GP with physically meaningful properties.
    """)


def render_conservative_fields():
    st.subheader("4.2 Conservative Force Fields")

    st.markdown("""
    One of the most fundamental constraints on any force field is **energy conservation**.
    Mathematically, we require that the curl vanishes everywhere:
    """)

    st.latex(r"\nabla \times \hat{G}[\hat{f}] = \nabla \times \mathbf{f_F} = \mathbf{0}")

    st.markdown("""
    The derivative operator $\\nabla$ satisfies this requirement. For energies and forces,
    we use the negative gradient operator:
    """)

    st.latex(r"\mathbf{\hat{f}_F(x)} = \hat{G}[\hat{f}] = -\nabla\hat{f}")

    st.success("""
    **Physical Meaning:** Forces are the negative gradient of the potential energy surface.
    By learning in the gradient domain, we guarantee energy conservation by construction.
    """)

    # Visualization of conservative vs non-conservative fields
    st.markdown("### Visualization: Conservative vs Non-Conservative Fields")

    col1, col2 = st.columns(2)

    with col1:
        st.markdown("**Conservative Field** (curl = 0)")
        fig1, ax1 = plt.subplots(figsize=(5, 5))

        # Conservative field: gradient of x^2 + y^2
        x = np.linspace(-2, 2, 10)
        y = np.linspace(-2, 2, 10)
        X, Y = np.meshgrid(x, y)
        U = -2 * X  # -dV/dx where V = x^2 + y^2
        V = -2 * Y  # -dV/dy

        ax1.quiver(X, Y, U, V, color='blue', alpha=0.7)
        ax1.set_xlabel('x')
        ax1.set_ylabel('y')
        ax1.set_title('F = -∇(x² + y²)')
        ax1.set_aspect('equal')
        ax1.grid(True, alpha=0.3)
        st.pyplot(fig1)
        plt.close()

    with col2:
        st.markdown("**Non-Conservative Field** (curl ≠ 0)")
        fig2, ax2 = plt.subplots(figsize=(5, 5))

        # Non-conservative field: rotational
        U = -Y
        V = X

        ax2.quiver(X, Y, U, V, color='red', alpha=0.7)
        ax2.set_xlabel('x')
        ax2.set_ylabel('y')
        ax2.set_title('F = (-y, x)')
        ax2.set_aspect('equal')
        ax2.grid(True, alpha=0.3)
        st.pyplot(fig2)
        plt.close()


def render_gdml_formulation():
    st.subheader("4.3 GDML Force Model")

    st.markdown("""
    Since differentiation is a linear operator, the result is another GP:
    """)

    st.latex(r"""
    \mathbf{\hat{f}_F(x)} \sim \mathcal{GP}\left[-\nabla_x\mu(x), \nabla_x k(x,x') \nabla_{x'}^T\right]
    """)

    st.markdown("The kernel becomes the Hessian of the original kernel:")

    st.latex(r"\nabla k \nabla^T = Hess_x(k)")
    st.latex(r"[Hess_x(k)]_{ij} = \frac{\partial^2 k}{\partial x_i \partial x_j}")

    st.markdown("""
    ### The GDML Force Estimator

    The trained force field collects contributions from partial derivatives of all training points:
    """)

    st.latex(r"""
    \mathbf{\hat{f}_F(x)} = \sum_i^M \sum_j^{3N} (\alpha_i)_j \frac{\partial}{\partial x_j} \nabla k(\mathbf{x}, \mathbf{x}_i)
    """)

    st.markdown("### Energy Predictor")

    st.markdown("""
    Because the trained model is a linear combination of kernel functions,
    integration only affects the kernel itself:
    """)

    st.latex(r"""
    \mathbf{\hat{f}_E(x)} = \sum_i^M \sum_j^{3N} (\alpha_i)_j \frac{\partial}{\partial x_j} k(\mathbf{x}, \mathbf{x}_i)
    """)

    st.success("""
    **Key Advantage:** GDML learns forces directly while guaranteeing energy conservation,
    and can predict both forces and energies from the same model.
    """)


def render_block_kernel():
    st.subheader("4.4 Block Kernel Construction")

    st.markdown("""
    ### Roto-translational Invariance

    Using a Coulomb-like matrix with inverse distances:
    """)

    st.latex(r"""
    D_{ij} = \begin{cases}
    \|r_i - r_j\|^{-1} & i > j \\
    0 & i \leq j
    \end{cases}
    """)

    st.markdown("Under the chain rule:")

    st.latex(r"\mathbf{k_F} = \nabla_x k_D \nabla_x^T = \mathbf{J_D}^T (\nabla_D k_D \nabla_D^T) \mathbf{J_D}")

    st.markdown("Where the Jacobian is:")

    st.latex(r"""
    \mathbf{J_D} = \begin{cases}
    \frac{\mathbf{r}_i - \mathbf{r}_j}{\|\mathbf{r}_i - \mathbf{r}_j\|^3} & i > j \\
    0 & i \leq j
    \end{cases}
    """)

    st.markdown("### The Matérn Kernel in GDML")

    st.latex(r"""
    k_{\nu=n+\frac{1}{2}}(d) = \exp\left(-\frac{\sqrt{2\nu}d}{\sigma}\right) P_n(d)
    """)

    st.latex(r"""
    P_n(d) = \sum_{k=0}^{n} \frac{(n+k)!}{(2n)!} \binom{n}{k} \left(\frac{2\sqrt{2\nu}d}{\sigma}\right)^{n-k}
    """)

    st.markdown("### Block Kernel Matrices")

    st.markdown("**Force Kernel:**")
    st.latex(r"\mathbf{k_F(x, x')} \in \mathbb{R}^{3N \times 3N}")

    st.markdown("**Energy Kernel:**")
    st.latex(r"\mathbf{k_E(x, x')} \in \mathbb{R}^{1 \times 3N}")

    # Interactive block kernel visualization
    st.markdown("### Block Kernel Visualization")

    n_atoms = st.slider("Number of atoms", 2, 6, 3, key="bk_atoms")

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    # Generate synthetic block kernel structure
    dim = 3 * n_atoms

    # Simulated force kernel structure (block diagonal dominant)
    K_F = np.zeros((dim, dim))
    for i in range(n_atoms):
        for j in range(n_atoms):
            block = np.eye(3) * np.exp(-0.5 * abs(i - j))
            if i != j:
                block += 0.3 * np.random.randn(3, 3)
            K_F[i*3:(i+1)*3, j*3:(j+1)*3] = block

    # Make symmetric
    K_F = (K_F + K_F.T) / 2

    # Force kernel heatmap
    im1 = axes[0].imshow(K_F, cmap='RdBu_r', aspect='auto')
    axes[0].set_title('Force Kernel $k_F$ Structure')
    axes[0].set_xlabel('Force component')
    axes[0].set_ylabel('Force component')

    # Add atom boundaries
    for i in range(1, n_atoms):
        axes[0].axhline(y=i*3-0.5, color='black', linewidth=0.5)
        axes[0].axvline(x=i*3-0.5, color='black', linewidth=0.5)

    plt.colorbar(im1, ax=axes[0])

    # Energy kernel visualization
    K_E = np.random.randn(1, dim) * 0.5
    K_E = np.abs(K_E)

    axes[1].bar(range(dim), K_E.ravel(), color='C0', alpha=0.7)
    axes[1].set_title('Energy Kernel $k_E$ Structure')
    axes[1].set_xlabel('Force component')
    axes[1].set_ylabel('Kernel value')

    # Add atom boundaries
    for i in range(1, n_atoms):
        axes[1].axvline(x=i*3-0.5, color='red', linewidth=1, linestyle='--')

    plt.tight_layout()
    st.pyplot(fig)
    plt.close()

    st.markdown("""
    ---

    ### Summary

    The GDML method provides:

    1. **Energy Conservation** - Forces derived from potential energy surface
    2. **Roto-translational Invariance** - Using distance-based descriptors
    3. **Physical Interpretability** - Clear relationship between energy and forces
    4. **Data Efficiency** - Physical constraints reduce required training data
    """)
