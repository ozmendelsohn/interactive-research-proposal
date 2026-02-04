"""
Data-Driven Force Fields for Large Scale Molecular Dynamics Simulations of Halide Perovskites
Main Streamlit Application
"""

import streamlit as st

st.set_page_config(
    page_title="GP Force Fields for MD Simulations",
    page_icon="🔬",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for better styling
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        font-weight: bold;
        color: #1f77b4;
        text-align: center;
        margin-bottom: 1rem;
    }
    .sub-header {
        font-size: 1.2rem;
        color: #666;
        text-align: center;
        margin-bottom: 2rem;
    }
    .stTabs [data-baseweb="tab-list"] {
        gap: 24px;
    }
    .stTabs [data-baseweb="tab"] {
        height: 50px;
        padding-left: 20px;
        padding-right: 20px;
    }
</style>
""", unsafe_allow_html=True)

# Main title
st.markdown('<p class="main-header">Data-Driven Force Fields for Molecular Dynamics</p>', unsafe_allow_html=True)
st.markdown('<p class="sub-header">Interactive Research Proposal: Gaussian Process Methods for Halide Perovskites</p>', unsafe_allow_html=True)

# Author info
st.sidebar.markdown("### Author")
st.sidebar.markdown("**Oz Yosef Mendelsohn**")
st.sidebar.markdown("ozyosef.mendelsohn@weizmann.ac.il")
st.sidebar.markdown("---")

# Navigation
st.sidebar.markdown("### Navigation")
page = st.sidebar.radio(
    "Select a section:",
    [
        "Introduction",
        "1. Normal Distributions",
        "2. Gaussian Processes",
        "3. GP Force Fields",
        "4. GDML Method",
        "About"
    ]
)

if page == "Introduction":
    st.markdown("""
    ## Halide Perovskites (HaPs)

    Halide perovskites (HaPs) are a class of materials with the general formula **ABX3**. HaPs achieve very
    high efficiency for solar cells in a very short amount of time while keeping relatively low manufacturing
    cost. Another added benefit of HaPs is the ability to manufacture them with a wide range of possible
    bandgap values, which can be used for many bandgap sensitive applications, including light-emitting
    diodes, photocatalysis, and more.
    """)

    col1, col2, col3 = st.columns([1, 2, 1])
    with col2:
        st.image("https://static.cambridge.org/binary/version/id/urn:cambridge.org:id:binary:20200701152913383-0351:S0883769420001402:S0883769420001402_fig1.png",
                 caption="Halide Perovskite Structure", width=600)

    st.markdown("""
    ### Theoretical Research of the Dynamics of Halide Perovskite

    Many of the properties of HaPs are based on the unique structure and dynamics of the HaPs system.
    One of the main methods of investigating the properties of HaPs theoretically is **molecular dynamics (MD)**.

    ### Molecular Dynamics (MD)

    Molecular Dynamics simulation is an algorithm for computing equilibrium and transport properties
    of many-body systems at a finite temperature. This method relies on the Newtonian equations of motion,
    updated each step numerically according to calculated forces.
    """)

    try:
        st.image("Images/MD.png", caption="Molecular Dynamics Simulation Process")
    except:
        st.info("MD diagram image not available")

    st.markdown("""
    ### Classical vs First-Principles MD

    | Method | Advantages | Limitations |
    |--------|-----------|-------------|
    | **Classical Potential MD** | Fast, scalable | Limited accuracy, requires empirical fitting |
    | **First-Principles MD (DFT)** | High accuracy, quantum effects | Computationally expensive |
    | **Data-Driven MD** | Combines accuracy with speed | Requires training data |

    ### Data-Driven Molecular Dynamics

    The ever-increasing processing power and the rise of shared data repositories provide the opportunity
    to easily obtain vast amounts of information for each desired system. This allows the emergence of
    **data-driven MD potentials** based on statistical learning.

    The two most common approaches are:
    - **Artificial Neural Networks (ANN)** - "Black box" but highly flexible
    - **Gaussian Processes (GP)** - More interpretable, can incorporate physical insights
    """)

    st.image("https://pubs.rsc.org/en/Image/Get?imageInfo.ImageType=GA&imageInfo.ImageIdentifier.ManuscriptID=C6SC05720A&imageInfo.ImageIdentifier.Year=2017",
             caption="Machine Learning for Molecular Dynamics")

elif page == "1. Normal Distributions":
    from pages import normal_distributions
    normal_distributions.render()

elif page == "2. Gaussian Processes":
    from pages import gaussian_processes
    gaussian_processes.render()

elif page == "3. GP Force Fields":
    from pages import gp_force_fields
    gp_force_fields.render()

elif page == "4. GDML Method":
    from pages import gdml_method
    gdml_method.render()

elif page == "About":
    st.markdown("""
    ## About This Project

    This interactive research proposal demonstrates the application of **Gaussian Process (GP)** and
    **Gradient-Domain Machine Learning (GDML)** methods to predict molecular forces in halide perovskite simulations.

    ### Key Concepts

    1. **Gaussian Process Regression** - Non-parametric Bayesian regression using kernel functions
    2. **GDML (Gradient-Domain Machine Learning)** - Learning force fields in the gradient domain for energy conservation
    3. **Periodic Boundary Conditions** - Handling atoms in periodic simulation cells
    4. **Roto-translational Invariance** - Using distance-based descriptors for physical consistency

    ### Technology Stack

    - **Core ML**: scikit-learn, SGDML
    - **Molecular Simulation**: ASE (Atomic Simulation Environment)
    - **Numerics**: NumPy, SciPy
    - **Visualization**: Matplotlib, Streamlit

    ### References

    - [SGDML Documentation](http://sgdml.org/)
    - [ASE Documentation](https://wiki.fysik.dtu.dk/ase/)
    - [Gaussian Processes for Machine Learning](http://www.gaussianprocess.org/gpml/)

    ---

    **Original Jupyter Notebook**: This Streamlit app was converted from an interactive Jupyter notebook
    designed to run in a Binder environment.
    """)

# Footer
st.sidebar.markdown("---")
st.sidebar.markdown("### Resources")
st.sidebar.markdown("[SGDML](http://sgdml.org/)")
st.sidebar.markdown("[ASE](https://wiki.fysik.dtu.dk/ase/)")
st.sidebar.markdown("[GP Tutorial](http://www.gaussianprocess.org/gpml/)")
