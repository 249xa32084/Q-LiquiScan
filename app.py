import streamlit as st
import pandas as pd
import numpy as np
import joblib
import os

# ============================================================
# PAGE CONFIGURATION
# ============================================================

st.set_page_config(
    page_title="Q-LiquiScan",
    page_icon="🧬",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ============================================================
# CUSTOM CSS
# ============================================================

st.markdown("""
<style>

    /* Main background */
    .stApp {
        background: linear-gradient(
            135deg,
            #07111f 0%,
            #0b1b2e 45%,
            #071522 100%
        );
        color: #f1f5f9;
    }

    /* Main content */
    .block-container {
        padding-top: 2rem;
        padding-bottom: 3rem;
        max-width: 1400px;
    }

    /* Sidebar */
    section[data-testid="stSidebar"] {
        background: linear-gradient(
            180deg,
            #081827 0%,
            #0d2438 100%
        );
        border-right: 1px solid #193b55;
    }

    section[data-testid="stSidebar"] h1,
    section[data-testid="stSidebar"] h2,
    section[data-testid="stSidebar"] h3,
    section[data-testid="stSidebar"] p,
    section[data-testid="stSidebar"] label {
        color: #e2e8f0 !important;
    }

    /* Main title */
    .main-title {
        font-size: 3rem;
        font-weight: 800;
        background: linear-gradient(
            90deg,
            #38bdf8,
            #818cf8,
            #c084fc
        );
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        margin-bottom: 0.2rem;
    }

    .subtitle {
        color: #94a3b8;
        font-size: 1.15rem;
        margin-bottom: 1.5rem;
    }

    /* Section titles */
    .section-title {
        font-size: 1.8rem;
        font-weight: 750;
        color: #e2e8f0;
        margin-top: 1rem;
        margin-bottom: 1rem;
    }

    /* Cards */
    .metric-card {
        background: linear-gradient(
            145deg,
            #102a40,
            #0b1e30
        );
        border: 1px solid #1e4863;
        border-radius: 18px;
        padding: 1.3rem;
        min-height: 140px;
        box-shadow: 0 8px 25px rgba(0,0,0,0.25);
    }

    .metric-icon {
        font-size: 1.7rem;
    }

    .metric-value {
        font-size: 2rem;
        font-weight: 800;
        color: #f8fafc;
        margin-top: 0.3rem;
    }

    .metric-label {
        color: #94a3b8;
        font-size: 0.95rem;
    }

    /* Information cards */
    .info-card {
        background: rgba(15, 42, 64, 0.75);
        border: 1px solid #1d4b66;
        border-radius: 16px;
        padding: 1.3rem;
        margin: 0.5rem 0;
    }

    .info-card h3 {
        color: #7dd3fc;
        margin-bottom: 0.5rem;
    }

    /* Pipeline */
    .pipeline {
        background: linear-gradient(
            135deg,
            #0b2438,
            #102d47
        );
        border: 1px solid #205273;
        border-radius: 18px;
        padding: 1.5rem;
        margin-top: 1rem;
    }

    .pipeline-step {
        text-align: center;
        padding: 0.8rem;
        color: #e2e8f0;
        font-weight: 600;
    }

    /* Result boxes */
    .result-success {
        background: linear-gradient(
            135deg,
            #063b32,
            #075743
        );
        border: 1px solid #10b981;
        border-radius: 18px;
        padding: 2rem;
        text-align: center;
        margin-top: 1rem;
    }

    .result-danger {
        background: linear-gradient(
            135deg,
            #451a1a,
            #641f1f
        );
        border: 1px solid #ef4444;
        border-radius: 18px;
        padding: 2rem;
        text-align: center;
        margin-top: 1rem;
    }

    .result-title {
        font-size: 1.8rem;
        font-weight: 800;
        margin-bottom: 0.5rem;
    }

    /* Small badge */
    .badge {
        display: inline-block;
        padding: 0.4rem 0.9rem;
        border-radius: 20px;
        background: #123b56;
        color: #7dd3fc;
        border: 1px solid #245b7b;
        font-size: 0.85rem;
        font-weight: 700;
        margin-right: 0.4rem;
    }

    /* Footer */
    .footer {
        text-align: center;
        color: #64748b;
        padding-top: 3rem;
        font-size: 0.85rem;
    }

    /* Buttons */
    .stButton > button {
        border-radius: 12px;
        font-weight: 700;
        min-height: 45px;
    }

</style>
""", unsafe_allow_html=True)


# ============================================================
# CONSTANTS
# ============================================================

DATA_FILE = "processed_data.csv"
MODEL_FILE = "classical_svm_model.pkl"

FEATURES = [
    "PC1",
    "PC2",
    "PC3",
    "PC4",
    "PC5",
    "PC6",
    "PC7",
    "PC8"
]


# ============================================================
# SESSION STATE
# ============================================================

if "demo_values" not in st.session_state:
    st.session_state.demo_values = [0.0] * 8

if "demo_name" not in st.session_state:
    st.session_state.demo_name = ""

if "prediction_result" not in st.session_state:
    st.session_state.prediction_result = None


# ============================================================
# SIDEBAR
# ============================================================

st.sidebar.markdown(
    """
    <div style="text-align:center;">
        <div style="font-size:3rem;">🧬</div>
        <h1 style="margin-bottom:0;">Q-LiquiScan</h1>
        <p style="color:#7dd3fc;">
            Quantum AI • Liquid Biopsy
        </p>
    </div>
    """,
    unsafe_allow_html=True
)

st.sidebar.markdown("---")

page = st.sidebar.radio(
    "🧭 Navigate",
    [
        "🏠 Dashboard",
        "📂 Dataset",
        "⚙️ Preprocessing",
        "⚛️ Quantum Model",
        "📊 Model Comparison",
        "🔍 Prediction",
        "📈 Results"
    ]
)

st.sidebar.markdown("---")

st.sidebar.markdown(
    """
    **Project Type**

    🧬 Bioinformatics  
    🤖 Machine Learning  
    ⚛️ Quantum Machine Learning  

    **Classification**

    Control vs Lung
    """
)


# ============================================================
# TOP HEADER
# ============================================================

st.markdown(
    '<div class="main-title">🧬 Q-LiquiScan</div>',
    unsafe_allow_html=True
)

st.markdown(
    '<div class="subtitle">'
    'Quantum SVM for Early Cancer Detection from Liquid Biopsy'
    '</div>',
    unsafe_allow_html=True
)

st.markdown(
    '<span class="badge">5hmC</span>'
    '<span class="badge">Machine Learning</span>'
    '<span class="badge">Quantum SVM</span>'
    '<span class="badge">Bioinformatics</span>',
    unsafe_allow_html=True
)

st.markdown("---")


# ============================================================
# DASHBOARD
# ============================================================

if page == "🏠 Dashboard":

    st.markdown(
        '<div class="section-title">🚀 Project Dashboard</div>',
        unsafe_allow_html=True
    )

    st.write(
        "An experimental AI platform for analyzing "
        "5hmC liquid-biopsy patterns."
    )

    # --------------------------------------------------------
    # METRICS
    # --------------------------------------------------------

    col1, col2, col3, col4 = st.columns(4)

    with col1:
        st.markdown("""
        <div class="metric-card">
            <div class="metric-icon">👥</div>
            <div class="metric-value">278</div>
            <div class="metric-label">Biological Samples</div>
        </div>
        """, unsafe_allow_html=True)

    with col2:
        st.markdown("""
        <div class="metric-card">
            <div class="metric-icon">🧬</div>
            <div class="metric-value">575,010</div>
            <div class="metric-label">Original Features</div>
        </div>
        """, unsafe_allow_html=True)

    with col3:
        st.markdown("""
        <div class="metric-card">
            <div class="metric-icon">📉</div>
            <div class="metric-value">8</div>
            <div class="metric-label">PCA Features</div>
        </div>
        """, unsafe_allow_html=True)

    with col4:
        st.markdown("""
        <div class="metric-card">
            <div class="metric-icon">⚛️</div>
            <div class="metric-value">8</div>
            <div class="metric-label">Quantum Qubits</div>
        </div>
        """, unsafe_allow_html=True)

    st.markdown("")

    # --------------------------------------------------------
    # PROJECT DESCRIPTION
    # --------------------------------------------------------

    col1, col2 = st.columns(2)

    with col1:

        st.markdown("""
        <div class="info-card">
            <h3>🧬 The Problem</h3>
            <p>
            Liquid-biopsy data can contain a very large number
            of genomic features. Identifying meaningful patterns
            from such high-dimensional data is computationally
            challenging.
            </p>
        </div>
        """, unsafe_allow_html=True)

    with col2:

        st.markdown("""
        <div class="info-card">
            <h3>💡 Our Approach</h3>
            <p>
            Q-LiquiScan combines dimensionality reduction,
            classical machine learning and quantum kernels to
            study Control vs Lung sample classification.
            </p>
        </div>
        """, unsafe_allow_html=True)

    # --------------------------------------------------------
    # PIPELINE
    # --------------------------------------------------------

    st.markdown(
        '<div class="section-title">🔬 Analysis Pipeline</div>',
        unsafe_allow_html=True
    )

    st.markdown("""
    <div class="pipeline">

    <div class="pipeline-step">
    🧬 Raw 5hmC Data
    </div>

    <div class="pipeline-step">
    ↓
    </div>

    <div class="pipeline-step">
    ⚙️ Preprocessing
    </div>

    <div class="pipeline-step">
    ↓
    </div>

    <div class="pipeline-step">
    📊 Feature Selection
    </div>

    <div class="pipeline-step">
    ↓
    </div>

    <div class="pipeline-step">
    📉 PCA → 8 Components
    </div>

    <div class="pipeline-step">
    ↓
    </div>

    <div class="pipeline-step">
    ⚛️ Quantum Feature Map
    </div>

    <div class="pipeline-step">
    ↓
    </div>

    <div class="pipeline-step">
    🤖 SVM Classification
    </div>

    </div>
    """, unsafe_allow_html=True)

    st.markdown("---")

    st.info(
        "⚠️ Q-LiquiScan is an experimental research/educational "
        "prototype and is not a clinical diagnostic system."
    )


# ============================================================
# DATASET
# ============================================================

elif page == "📂 Dataset":

    st.markdown(
        '<div class="section-title">📂 Dataset Center</div>',
        unsafe_allow_html=True
    )

    st.write(
        "View the processed dataset used by the machine-learning models."
    )

    if os.path.exists(DATA_FILE):

        data = pd.read_csv(DATA_FILE)

        # ----------------------------------------------------
        # DATASET METRICS
        # ----------------------------------------------------

        total_samples = data.shape[0]
        total_columns = data.shape[1]
        total_features = total_columns - 1

        control_count = int(
            (data["Label"] == 0).sum()
        )

        lung_count = int(
            (data["Label"] == 1).sum()
        )

        col1, col2, col3, col4 = st.columns(4)

        with col1:
            st.markdown(f"""
            <div class="metric-card">
                <div class="metric-icon">👥</div>
                <div class="metric-value">{total_samples}</div>
                <div class="metric-label">Total Samples</div>
            </div>
            """, unsafe_allow_html=True)

        with col2:
            st.markdown(f"""
            <div class="metric-card">
                <div class="metric-icon">🧬</div>
                <div class="metric-value">{total_features}</div>
                <div class="metric-label">PCA Features</div>
            </div>
            """, unsafe_allow_html=True)

        with col3:
            st.markdown(f"""
            <div class="metric-card">
                <div class="metric-icon">🟢</div>
                <div class="metric-value">{control_count}</div>
                <div class="metric-label">Control Samples</div>
            </div>
            """, unsafe_allow_html=True)

        with col4:
            st.markdown(f"""
            <div class="metric-card">
                <div class="metric-icon">🔴</div>
                <div class="metric-value">{lung_count}</div>
                <div class="metric-label">Lung Samples</div>
            </div>
            """, unsafe_allow_html=True)

        st.markdown("")

        # ----------------------------------------------------
        # ORIGINAL VS PROCESSED
        # ----------------------------------------------------

        st.markdown(
            '<div class="section-title">'
            '📊 Original vs Processed Data'
            '</div>',
            unsafe_allow_html=True
        )

        comparison = pd.DataFrame({
            "Stage": [
                "Original Dataset",
                "After Variance Filtering",
                "Feature Selection",
                "Final PCA Dataset"
            ],
            "Features": [
                "575,010",
                "540,345",
                "500",
                "8"
            ],
            "Purpose": [
                "Raw genomic data",
                "Remove constant features",
                "Keep highly variable features",
                "Reduce dimensionality"
            ]
        })

        st.dataframe(
            comparison,
            width="stretch",
            hide_index=True
        )

        st.info(
            "The number of biological samples remains 278. "
            "Only the number of features is reduced."
        )

        # ----------------------------------------------------
        # FULL DATASET
        # ----------------------------------------------------

        st.markdown(
            '<div class="section-title">'
            '📋 Complete Processed Dataset'
            '</div>',
            unsafe_allow_html=True
        )

        st.write(
            f"Showing {total_samples} samples and "
            f"{total_columns} columns."
        )

        st.dataframe(
            data,
            width="stretch",
            height=550
        )

    else:

        st.error(
            "processed_data.csv was not found."
        )

        st.info(
            "Run preprocess_data.py first."
        )


# ============================================================
# PREPROCESSING
# ============================================================

elif page == "⚙️ Preprocessing":

    st.markdown(
        '<div class="section-title">⚙️ Data Preprocessing</div>',
        unsafe_allow_html=True
    )

    st.write(
        "The preprocessing pipeline converts high-dimensional "
        "5hmC data into a compact representation suitable for "
        "classical and quantum machine learning."
    )

    steps = [
        (
            "1️⃣",
            "Remove Metadata",
            "Remove Geneid, chromosome, start, end, strand and length."
        ),
        (
            "2️⃣",
            "Variance Filtering",
            "Remove features with zero variance."
        ),
        (
            "3️⃣",
            "Feature Selection",
            "Select the top 500 highly variable features."
        ),
        (
            "4️⃣",
            "Standardization",
            "Scale the selected features."
        ),
        (
            "5️⃣",
            "PCA",
            "Reduce 500 features to 8 principal components."
        )
    ]

    for icon, title, description in steps:

        st.markdown(
            f"""
            <div class="info-card">
                <h3>{icon} {title}</h3>
                <p>{description}</p>
            </div>
            """,
            unsafe_allow_html=True
        )

    st.markdown("---")

    col1, col2, col3 = st.columns(3)

    with col1:
        st.metric(
            "Original Features",
            "575,010"
        )

    with col2:
        st.metric(
            "Selected Features",
            "500"
        )

    with col3:
        st.metric(
            "PCA Components",
            "8"
        )

    st.success(
        "The current 8 PCA components explain approximately "
        "97.03% of the variance."
    )


# ============================================================
# QUANTUM MODEL
# ============================================================

elif page == "⚛️ Quantum Model":

    st.markdown(
        '<div class="section-title">⚛️ Quantum SVM</div>',
        unsafe_allow_html=True
    )

    st.write(
        "The Quantum Support Vector Machine uses a quantum "
        "kernel to measure similarity between samples."
    )

    col1, col2, col3, col4 = st.columns(4)

    with col1:
        st.metric("Qubits", "8")

    with col2:
        st.metric("Features", "8")

    with col3:
        st.metric("Feature Map", "ZZ")

    with col4:
        st.metric("Kernel", "Fidelity")

    st.markdown("---")

    st.markdown("""
    <div class="info-card">
        <h3>⚛️ Quantum Feature Map</h3>
        <p>
        The 8 PCA components are encoded into an 8-qubit
        quantum feature map using a ZZ-based circuit.
        </p>
    </div>
    """, unsafe_allow_html=True)

    st.markdown("""
    <div class="info-card">
        <h3>🔗 Fidelity Quantum Kernel</h3>
        <p>
        The quantum kernel evaluates the similarity between
        quantum states generated from different samples.
        </p>
    </div>
    """, unsafe_allow_html=True)

    st.markdown("""
    <div class="info-card">
        <h3>🤖 Quantum SVM Classifier</h3>
        <p>
        The resulting kernel matrix is supplied to an SVM
        classifier for Control vs Lung classification.
        </p>
    </div>
    """, unsafe_allow_html=True)


# ============================================================
# MODEL COMPARISON
# ============================================================

elif page == "📊 Model Comparison":

    st.markdown(
        '<div class="section-title">'
        '📊 Classical SVM vs Quantum SVM'
        '</div>',
        unsafe_allow_html=True
    )

    st.write(
        "Both models use the same processed data and the same "
        "80:20 stratified train-test split."
    )

    comparison = pd.DataFrame({
        "Metric": [
            "Accuracy",
            "Precision",
            "Lung Recall",
            "F1 Score",
            "Balanced Accuracy"
        ],
        "Classical SVM": [
            "78.57%",
            "33.33%",
            "9.09%",
            "14.29%",
            "—"
        ],
        "Quantum SVM": [
            "80.36%",
            "0.00%",
            "0.00%",
            "0.00%",
            "50.00%"
        ]
    })

    st.dataframe(
        comparison,
        width="stretch",
        hide_index=True
    )

    st.markdown("---")

    col1, col2 = st.columns(2)

    with col1:

        st.markdown("""
        <div class="info-card">
            <h3>🤖 Classical SVM</h3>
            <p>
            Uses an RBF kernel to classify the PCA feature vectors.
            </p>
            <h3>78.57%</h3>
            <p>Overall accuracy</p>
        </div>
        """, unsafe_allow_html=True)

    with col2:

        st.markdown("""
        <div class="info-card">
            <h3>⚛️ Quantum SVM</h3>
            <p>
            Uses a fidelity quantum kernel with an 8-qubit
            feature map.
            </p>
            <h3>80.36%</h3>
            <p>Overall accuracy</p>
        </div>
        """, unsafe_allow_html=True)

    st.warning("""
    Important: The current QSVM achieved 80.36% accuracy but
    predicted every test sample as Control. Therefore, its
    Lung recall and F1 score were 0%. Accuracy alone should not
    be used to judge this imbalanced classification problem.
    """)


# ============================================================
# PREDICTION
# ============================================================

elif page == "🔍 Prediction":

    st.markdown(
        '<div class="section-title">'
        '🔍 Cancer Pattern Prediction'
        '</div>',
        unsafe_allow_html=True
    )

    st.write(
        "Enter PCA values manually or load a real sample from "
        "the processed dataset."
    )

    # --------------------------------------------------------
    # CHECK FILES
    # --------------------------------------------------------

    if not os.path.exists(DATA_FILE):

        st.error("processed_data.csv was not found.")
        st.stop()

    if not os.path.exists(MODEL_FILE):

        st.error("classical_svm_model.pkl was not found.")
        st.stop()

    # --------------------------------------------------------
    # LOAD DATA
    # --------------------------------------------------------

    data = pd.read_csv(DATA_FILE)

    # Load the balanced SVM model
    model = joblib.load(MODEL_FILE)

    # --------------------------------------------------------
    # DEMO SAMPLES
    # --------------------------------------------------------

    st.markdown(
        '<div class="section-title">🎯 Demo Samples</div>',
        unsafe_allow_html=True
    )

    st.write(
        "Use actual samples from the processed dataset "
        "for demonstration."
    )

    col1, col2, col3 = st.columns(3)

    # --------------------------------------------------------
    # CONTROL DEMO
    # --------------------------------------------------------

    with col1:

        if st.button(
            "🟢 Load Control Demo",
            width="stretch"
        ):

            control_rows = data[data["Label"] == 0]

            if len(control_rows) > 0:

                row = control_rows.iloc[0]

                st.session_state.demo_values = [
                    float(row[feature])
                    for feature in FEATURES
                ]

                st.session_state.demo_name = "Control Demo"
                st.session_state.prediction_result = None

                st.rerun()

    # --------------------------------------------------------
    # LUNG DEMO
    # --------------------------------------------------------

    with col2:

        if st.button(
            "🔴 Load Lung Demo",
            width="stretch"
        ):

            lung_rows = data[data["Label"] == 1]

            if len(lung_rows) > 0:

                row = lung_rows.iloc[0]

                st.session_state.demo_values = [
                    float(row[feature])
                    for feature in FEATURES
                ]

                st.session_state.demo_name = "Lung Demo"
                st.session_state.prediction_result = None

                st.rerun()

    # --------------------------------------------------------
    # CLEAR
    # --------------------------------------------------------

    with col3:

        if st.button(
            "🧹 Clear All",
            width="stretch"
        ):

            st.session_state.demo_values = [0.0] * 8
            st.session_state.demo_name = ""
            st.session_state.prediction_result = None

            st.rerun()

    if st.session_state.demo_name:

        st.success(
            f"Currently loaded: {st.session_state.demo_name}"
        )

    # --------------------------------------------------------
    # PCA INPUT
    # --------------------------------------------------------

    st.markdown(
        '<div class="section-title">🧬 PCA Feature Input</div>',
        unsafe_allow_html=True
    )

    col1, col2 = st.columns(2)

    with col1:

        pc1 = st.number_input(
            "PC1",
            value=float(st.session_state.demo_values[0]),
            format="%.6f"
        )

        pc2 = st.number_input(
            "PC2",
            value=float(st.session_state.demo_values[1]),
            format="%.6f"
        )

        pc3 = st.number_input(
            "PC3",
            value=float(st.session_state.demo_values[2]),
            format="%.6f"
        )

        pc4 = st.number_input(
            "PC4",
            value=float(st.session_state.demo_values[3]),
            format="%.6f"
        )

    with col2:

        pc5 = st.number_input(
            "PC5",
            value=float(st.session_state.demo_values[4]),
            format="%.6f"
        )

        pc6 = st.number_input(
            "PC6",
            value=float(st.session_state.demo_values[5]),
            format="%.6f"
        )

        pc7 = st.number_input(
            "PC7",
            value=float(st.session_state.demo_values[6]),
            format="%.6f"
        )

        pc8 = st.number_input(
            "PC8",
            value=float(st.session_state.demo_values[7]),
            format="%.6f"
        )

    pc_values = [
        pc1,
        pc2,
        pc3,
        pc4,
        pc5,
        pc6,
        pc7,
        pc8
    ]

    st.markdown("---")

    # --------------------------------------------------------
    # ANALYZE
    # --------------------------------------------------------

    col1, col2 = st.columns(2)

    with col1:

        if st.button(
            "🔍 ANALYZE SAMPLE",
            width="stretch"
        ):

            input_data = pd.DataFrame(
                [pc_values],
                columns=FEATURES
            )

            prediction = model.predict(input_data)

            predicted_class = int(prediction[0])

            st.session_state.prediction_result = predicted_class

    # --------------------------------------------------------
    # CLEAR PREDICTION
    # --------------------------------------------------------

    with col2:

        if st.button(
            "🧹 CLEAR PREDICTION",
            width="stretch"
        ):

            st.session_state.demo_values = [0.0] * 8
            st.session_state.demo_name = ""
            st.session_state.prediction_result = None

            st.rerun()

    # --------------------------------------------------------
    # RESULT
    # --------------------------------------------------------

    if st.session_state.prediction_result is not None:

        predicted_class = st.session_state.prediction_result

        st.markdown("---")

        st.markdown(
            '<div class="section-title">'
            '📋 Analysis Result'
            '</div>',
            unsafe_allow_html=True
        )

        # ----------------------------------------------------
        # CONTROL RESULT
        # ----------------------------------------------------

        if predicted_class == 0:

            st.success(
                "🟢 NO CANCER-LIKE PATTERN DETECTED\n\n"
                "The model classified this sample as Control.\n\n"
                "The sample shows a Control-like pattern "
                "according to the current model."
            )

            st.info(
                "Model interpretation: Lower-risk pattern "
                "relative to the Lung class."
            )

        # ----------------------------------------------------
        # LUNG RESULT
        # ----------------------------------------------------

        else:

            st.error(
                "🔴 CANCER-LIKE PATTERN DETECTED\n\n"
                "The model classified this sample as Lung.\n\n"
                "The sample shows a Lung-like pattern "
                "according to the current model."
            )

            st.warning(
                "Model interpretation: Higher-risk pattern "
                "relative to the Control class."
            )

        # ----------------------------------------------------
        # MEDICAL DISCLAIMER
        # ----------------------------------------------------

        st.markdown("---")

        st.warning(
            """
            ⚠️ IMPORTANT MEDICAL DISCLAIMER

            This prediction is generated by an experimental
            machine-learning model for educational and research
            purposes.

            It is NOT a medical diagnosis.

            A positive prediction does not confirm cancer, and a
            negative prediction does not rule out cancer.

            Please consult a qualified doctor or healthcare
            professional for proper medical evaluation.
            """
        )

        st.info(
            "💬 AI can assist with data analysis, but medical "
            "decisions should always be made with qualified "
            "healthcare professionals."
        )

# ============================================================
# RESULTS
# ============================================================

elif page == "📈 Results":

    st.markdown(
        '<div class="section-title">📈 Experimental Results</div>',
        unsafe_allow_html=True
    )

    st.write(
        "Current performance obtained from the experimental "
        "80:20 stratified evaluation."
    )

    # --------------------------------------------------------
    # METRICS
    # --------------------------------------------------------

    col1, col2 = st.columns(2)

    with col1:

        st.markdown("""
        <div class="info-card">
            <h3>🤖 Classical SVM</h3>
            <p>Accuracy</p>
            <h2>78.57%</h2>
            <p>Lung Recall: 9.09%</p>
            <p>F1 Score: 14.29%</p>
        </div>
        """, unsafe_allow_html=True)

    with col2:

        st.markdown("""
        <div class="info-card">
            <h3>⚛️ Quantum SVM</h3>
            <p>Accuracy</p>
            <h2>80.36%</h2>
            <p>Lung Recall: 0.00%</p>
            <p>F1 Score: 0.00%</p>
        </div>
        """, unsafe_allow_html=True)

    # --------------------------------------------------------
    # CONFUSION MATRICES
    # --------------------------------------------------------

    st.markdown(
        '<div class="section-title">🔢 Confusion Matrices</div>',
        unsafe_allow_html=True
    )

    col1, col2 = st.columns(2)

    with col1:

        st.subheader("🤖 Classical SVM")

        classical_cm = pd.DataFrame(
            [
                [43, 2],
                [10, 1]
            ],
            index=[
                "Actual Control",
                "Actual Lung"
            ],
            columns=[
                "Predicted Control",
                "Predicted Lung"
            ]
        )

        st.dataframe(
            classical_cm,
            width="stretch"
        )

    with col2:

        st.subheader("⚛️ Quantum SVM")

        quantum_cm = pd.DataFrame(
            [
                [45, 0],
                [11, 0]
            ],
            index=[
                "Actual Control",
                "Actual Lung"
            ],
            columns=[
                "Predicted Control",
                "Predicted Lung"
            ]
        )

        st.dataframe(
            quantum_cm,
            width="stretch"
        )

    st.markdown("---")

    st.warning("""
    The current QSVM predicts all test samples as Control.
    Therefore, although the overall accuracy is 80.36%, the
    model does not currently detect the Lung class effectively.
    """)


# ============================================================
# FOOTER
# ============================================================

st.markdown(
    """
    <div class="footer">
        🧬 Q-LiquiScan &nbsp;|&nbsp;
        Quantum Machine Learning for Liquid Biopsy Research
        <br><br>
        Experimental / Educational Prototype — Not for Clinical Diagnosis
    </div>
    """,
    unsafe_allow_html=True
)

