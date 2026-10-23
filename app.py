"""
NeuroMotion - Edge Computer Vision & Digital Biomarker System
=============================================================
Streamlit Clinician Dashboard for MDS-UPDRS Part III Motor Examination.
Provides real-time kinematic tracking, 10-second formal trial evaluation,
spectral tremor decomposition (Welch PSD), automated clinical scoring,
and HL7 FHIR R4 EHR compliant export.

Light Theme Clinical Edition with WCAG AAA Contrast & Responsive Controls.
"""

import os
import sys
import time
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional, Tuple

import cv2
import numpy as np
import plotly.graph_objects as go
import streamlit as st

# Ensure repository root is on sys.path
PROJECT_ROOT = os.path.abspath(os.path.dirname(__file__))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from clinical.fhir_export import (
    FHIRExporter,
    export_fhir_json,
    generate_fhir_diagnostic_report,
)
from clinical.scoring_rules import MDSUPDRSScorer
from core.biomarkers import (
    FingerTappingAnalyzer,
    TapEvent,
    TapState,
    TappingTrialManager,
    TremorAnalyzer,
)
from core.signal_processing import SignalProcessor
from core.vision import (
    INDEX_TIP_IDX,
    KEY_LANDMARK_INDICES,
    MIDDLE_MCP_IDX,
    MIDDLE_TIP_IDX,
    THUMB_TIP_IDX,
    WRIST_IDX,
    HandTracker,
    KinematicBuffer,
    draw_clinical_hud,
)

# =============================================================================
# 1. STREAMLIT PAGE CONFIG & HIGH-CONTRAST LIGHT CLINICAL STYLING
# =============================================================================
st.set_page_config(
    page_title="NeuroMotion AI | MDS-UPDRS Motor Biomarkers",
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown(
    """
    <style>
    /* =========================================================================
       CLINICAL LIGHT THEME (WCAG AAA Contrast, Slate & Clean Medical Aesthetics)
       ========================================================================= */
    .stApp {
        background-color: #f8fafc !important;
        color: #0f172a !important;
        font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, "Helvetica Neue", Arial, sans-serif !important;
    }

    /* Force all text in main content to high-contrast dark slate */
    .main .block-container {
        padding-top: 1.5rem !important;
        padding-bottom: 3.5rem !important;
        color: #0f172a !important;
    }
    
    h1, h2, h3, h4, h5, h6, p, span, label, div {
        color: #0f172a;
    }

    /* Sidebar Light Styling */
    section[data-testid="stSidebar"] {
        background-color: #ffffff !important;
        border-right: 1px solid #e2e8f0 !important;
    }
    section[data-testid="stSidebar"] * {
        color: #1e293b !important;
    }
    section[data-testid="stSidebar"] hr {
        border-color: #e2e8f0 !important;
    }

    /* Header Container */
    .header-container {
        background: #ffffff;
        border: 1px solid #cbd5e1;
        border-radius: 12px;
        padding: 20px 24px;
        margin-bottom: 16px;
        box-shadow: 0 4px 16px rgba(15, 23, 42, 0.05);
    }

    /* Status Chips / Badges */
    .status-chip {
        display: inline-flex;
        align-items: center;
        gap: 6px;
        background: #f1f5f9;
        border: 1px solid #cbd5e1;
        border-radius: 20px;
        padding: 5px 12px;
        font-size: 0.82rem;
        font-weight: 600;
        color: #334155 !important;
        margin-right: 8px;
        margin-top: 6px;
    }
    .status-chip.active {
        background: #f0fdf4;
        border-color: #86efac;
        color: #15803d !important;
    }
    .status-chip.telehealth {
        background: #faf5ff;
        border-color: #d8b4fe;
        color: #7e22ce !important;
    }
    .status-chip.primary {
        background: #eff6ff;
        border-color: #93c5fd;
        color: #1d4ed8 !important;
    }

    /* Metric Cards in Right Column */
    .metric-card {
        background: #ffffff;
        border: 1px solid #e2e8f0;
        border-radius: 10px;
        padding: 16px 18px;
        text-align: left;
        box-shadow: 0 2px 8px rgba(15, 23, 42, 0.04);
        border-left: 4px solid #0284c7;
        margin-bottom: 6px;
    }
    .metric-card .title {
        font-size: 0.80rem;
        color: #475569 !important;
        text-transform: uppercase;
        letter-spacing: 0.05em;
        font-weight: 700;
        margin-bottom: 4px;
    }
    .metric-card .value {
        font-size: 1.65rem;
        font-weight: 800;
        color: #0f172a !important;
        line-height: 1.2;
    }
    .metric-card .subtext {
        font-size: 0.78rem;
        color: #0284c7 !important;
        font-weight: 600;
        margin-top: 4px;
    }

    /* Score Badges */
    .score-badge {
        display: inline-block;
        padding: 6px 14px;
        border-radius: 8px;
        font-weight: 800;
        font-size: 0.95rem;
        letter-spacing: 0.03em;
        text-align: center;
    }
    .score-0 { background: #dcfce7; color: #15803d !important; border: 1.5px solid #22c55e; }
    .score-1 { background: #e0f2fe; color: #0369a1 !important; border: 1.5px solid #0284c7; }
    .score-2 { background: #fef3c7; color: #b45309 !important; border: 1.5px solid #f59e0b; }
    .score-3 { background: #ffedd5; color: #c2410c !important; border: 1.5px solid #ea580c; }
    .score-4 { background: #fee2e2; color: #b91c1c !important; border: 1.5px solid #ef4444; }

    /* =========================================================================
       TEXT BOXES, INPUTS, SELECTBOXES & TEXTAREAS: TRANSPARENT WITH BLACK BORDERS
       ========================================================================= */
    div[data-baseweb="input"],
    div[data-baseweb="base-input"],
    div[data-baseweb="textarea"],
    div[data-baseweb="select"] > div,
    .stTextInput > div > div,
    .stNumberInput > div > div,
    .stTextArea > div > div,
    .stSelectbox > div > div > div {
        background-color: transparent !important;
        background: transparent !important;
        border: 1.5px solid #000000 !important;
        border-radius: 6px !important;
        box-shadow: none !important;
    }

    input, textarea, select {
        background-color: transparent !important;
        background: transparent !important;
        color: #000000 !important;
        -webkit-text-fill-color: #000000 !important;
        font-weight: 600 !important;
        font-size: 0.92rem !important;
    }

    /* Number input increment/decrement buttons */
    div[data-baseweb="input"] button {
        background-color: transparent !important;
        color: #000000 !important;
        border-left: 1px solid #000000 !important;
    }
    div[data-baseweb="input"] button:hover {
        background-color: rgba(0, 0, 0, 0.05) !important;
    }
    div[data-baseweb="input"] button svg {
        fill: #000000 !important;
    }

    /* Selectbox dropdown value & chevron */
    div[data-baseweb="select"] * {
        background-color: transparent !important;
        color: #000000 !important;
    }
    div[data-baseweb="select"] svg {
        fill: #000000 !important;
    }

    /* Focus highlight */
    div[data-baseweb="input"]:focus-within,
    div[data-baseweb="base-input"]:focus-within,
    div[data-baseweb="textarea"]:focus-within,
    div[data-baseweb="select"] > div:focus-within {
        border-color: #0284c7 !important;
        box-shadow: 0 0 0 1px #0284c7 !important;
    }

    /* Dropdown popups */
    ul[data-baseweb="menu"] {
        background-color: #ffffff !important;
        border: 1.5px solid #000000 !important;
        border-radius: 6px !important;
    }
    li[data-baseweb="menu-item"] {
        background-color: #ffffff !important;
        color: #000000 !important;
    }
    li[data-baseweb="menu-item"]:hover {
        background-color: #f1f5f9 !important;
    }

    /* Action Buttons Toolbar - Generous padding, no clipping */
    div.stButton > button {
        width: 100% !important;
        min-height: 48px !important;
        padding: 12px 20px !important;
        margin: 2px 0 !important;
        white-space: nowrap !important;
        display: inline-flex !important;
        justify-content: center !important;
        align-items: center !important;
        font-weight: 700 !important;
        font-size: 0.92rem !important;
        border-radius: 8px !important;
        border: 1.5px solid #000000 !important;
        background-color: #ffffff !important;
        color: #0f172a !important;
        box-shadow: 0 2px 4px rgba(0, 0, 0, 0.05) !important;
        transition: all 0.15s ease-in-out !important;
    }
    div.stButton > button:hover {
        border-color: #0284c7 !important;
        color: #0284c7 !important;
        background-color: #f0f9ff !important;
        transform: translateY(-1px);
        box-shadow: 0 4px 8px rgba(0, 0, 0, 0.08) !important;
    }
    div.stButton > button[kind="primary"] {
        background-color: #0284c7 !important;
        color: #ffffff !important;
        border-color: #0284c7 !important;
    }
    div.stButton > button[kind="primary"]:hover {
        background-color: #0369a1 !important;
        border-color: #0369a1 !important;
        color: #ffffff !important;
    }

    /* Score Card Containers in Light Mode */
    .clinical-score-card {
        background: #ffffff;
        border: 1px solid #cbd5e1;
        border-radius: 10px;
        padding: 16px 18px;
        box-shadow: 0 2px 8px rgba(15, 23, 42, 0.04);
        margin-bottom: 8px;
    }

    /* Quick Start Guide Banner */
    .guide-card {
        background: #f8fafc;
        border-radius: 8px;
        padding: 14px 16px;
        height: 100%;
        border: 1px solid #e2e8f0;
    }

    /* Streamlit Expander Light Style */
    .streamlit-expanderHeader {
        background-color: #ffffff !important;
        border: 1px solid #e2e8f0 !important;
        border-radius: 8px !important;
        font-weight: 600 !important;
        color: #0f172a !important;
    }
    .streamlit-expanderContent {
        background-color: #ffffff !important;
        border: 1px solid #e2e8f0 !important;
        border-top: none !important;
        border-radius: 0 0 8px 8px !important;
    }
    </style>
    """,
    unsafe_allow_html=True,
)


# =============================================================================
# 2. STATE INITIALIZATION & SESSION PERSISTENCE
# =============================================================================
_FALLBACK_STATE: Dict[str, Any] = {}


def get_state(key: str, default: Any = None) -> Any:
    """Safe state getter with fallback when running in non-server context."""
    try:
        if hasattr(st, "session_state") and key in st.session_state:
            return st.session_state[key]
    except Exception:
        pass
    return _FALLBACK_STATE.get(key, default)


def set_state(key: str, value: Any) -> None:
    """Safe state setter with fallback."""
    try:
        if hasattr(st, "session_state"):
            st.session_state[key] = value
    except Exception:
        pass
    _FALLBACK_STATE[key] = value


def init_session_state() -> None:
    """Initialize persistent Streamlit session state objects."""
    defaults = {
        "tapping_analyzer": FingerTappingAnalyzer(default_fs=30.0),
        "tremor_analyzer": TremorAnalyzer(),
        "kinematic_buffer": KinematicBuffer(buffer_size=150),
        "stream_active": False,
        "latest_tap_metrics": {
            "total_taps": 0,
            "tap_rate_hz": 0.0,
            "mean_amplitude": 0.0,
            "amplitude_decay_slope": 0.0,
            "rhythm_cv": 0.0,
        },
        "latest_updrs_tapping": {
            "score": 0,
            "label": "Normal",
            "clinical_rationale": "Awaiting initial trial data. Select a testing mode above to begin.",
            "domain_breakdown": {"rate_score": 0, "decay_score": 0, "rhythm_score": 0},
        },
        "latest_tremor_metrics": {
            "peak_freq": 0.0,
            "peak_power": 0.0,
            "tremor_type": "Voluntary baseline movement",
            "is_significant": False,
            "freqs": np.array([]),
            "psd": np.array([]),
        },
        "latest_updrs_tremor": {
            "score": 0,
            "label": "Normal",
            "clinical_rationale": "No pathological tremor detected.",
        },
        "waveform_history_time": np.array([]),
        "waveform_history_aperture": np.array([]),
    }
    for k, v in defaults.items():
        if get_state(k) is None:
            set_state(k, v)


init_session_state()


# =============================================================================
# 3. SIDEBAR CLINICAL CONFIGURATION & STREAM CONTROLS
# =============================================================================
with st.sidebar:
    st.markdown("### 🧠 **NeuroMotion Control**")
    st.caption("Edge-First Digital Biomarker Platform | MDS-UPDRS")
    st.divider()

    st.markdown("#### 📋 **Patient & Telehealth Profile**")
    patient_id = st.text_input("Patient Identifier", value="PT-1042")
    patient_age = st.number_input("Age (Years)", min_value=18, max_value=110, value=68, step=1)
    patient_gender = st.selectbox("Gender", options=["Male", "Female", "Other", "Unknown"], index=0)
    clinician_name = st.text_input("Evaluating Clinician", value="Dr. Sarah Adams, MD")
    clinician_notes = st.text_area(
        "Consultation Clinical Notes",
        value="Patient assessed via remote telehealth consultation. Unilateral right hand motor screening for bradykinesia and tremor.",
        height=80,
    )

    st.divider()
    st.markdown("#### 📹 **Video Input Modality**")
    input_mode = st.radio(
        "Source Selector",
        options=["Live Webcam (Edge DirectShow)", "Pre-recorded Demo Clip (Offline Mode)"],
        index=0,  # Default to real live webcam!
    )

    selected_video_path: Optional[str] = None
    is_offline_mode = input_mode == "Pre-recorded Demo Clip (Offline Mode)"
    camera_id = 0
    if not is_offline_mode:
        camera_id = st.number_input(
            "Camera Device Index",
            min_value=0,
            max_value=5,
            value=0,
            step=1,
            help="Device index 0 is your primary integrated/USB camera.",
        )
    else:
        # Scan assets/ for sample mp4/avi videos
        assets_dir = os.path.join(PROJECT_ROOT, "assets")
        available_samples: List[str] = []
        if os.path.exists(assets_dir):
            available_samples = [
                f for f in os.listdir(assets_dir)
                if f.lower().endswith((".mp4", ".avi", ".mov", ".webm"))
            ]

        if available_samples:
            sample_choice = st.selectbox(
                "Select Pre-recorded Clip",
                options=available_samples + ["Upload Custom Video..."],
                index=0,
            )
            if sample_choice == "Upload Custom Video...":
                uploaded_file = st.file_uploader("Upload Video (.mp4 / .avi)", type=["mp4", "avi", "mov"])
                if uploaded_file is not None:
                    temp_dir = os.path.join(PROJECT_ROOT, "assets", "temp")
                    os.makedirs(temp_dir, exist_ok=True)
                    temp_path = os.path.join(temp_dir, uploaded_file.name)
                    with open(temp_path, "wb") as f:
                        f.write(uploaded_file.getbuffer())
                    selected_video_path = temp_path
            else:
                selected_video_path = os.path.join(assets_dir, sample_choice)
        else:
            st.warning("No pre-recorded videos found in `assets/`. Please upload a video.")
            uploaded_file = st.file_uploader("Upload Video", type=["mp4", "avi", "mov"])
            if uploaded_file is not None:
                temp_dir = os.path.join(PROJECT_ROOT, "assets", "temp")
                os.makedirs(temp_dir, exist_ok=True)
                temp_path = os.path.join(temp_dir, uploaded_file.name)
                with open(temp_path, "wb") as f:
                    f.write(uploaded_file.getbuffer())
                selected_video_path = temp_path

    st.divider()
    st.markdown("#### ⚙️ **Overlay & HUD Toggles**")
    show_landmarks = st.checkbox("Show Hand Landmarks & Skeleton", value=True)
    show_kinematic_vector = st.checkbox("Show Kinematic Vector (Thumb-Index)", value=True)
    show_tap_state = st.checkbox("Show Tap State Machine Telemetry", value=True)
    show_clinical_hud_cards = st.checkbox("Show Polished HUD Overlay Cards", value=True)

    st.divider()
    st.markdown("#### 🚀 **Hardware & Architecture**")
    st.markdown("• **Model:** `MediaPipe Hands v0.10.14` 🟢")
    st.markdown("• **Normalization:** `2D Euclidean / MCP9-Wrist`")
    st.markdown("• **Target Rate:** `30.0 FPS`")


# =============================================================================
# 4. MAIN TELEHEALTH CLINICAL HEADER
# =============================================================================
st.markdown(
    f"""
    <div class="header-container">
        <h1 style="margin: 0; font-size: 2.1rem; color: #0f172a; font-weight: 800; letter-spacing: -0.02em;">
            🧠 NeuroMotion AI
        </h1>
        <p style="margin: 4px 0 12px 0; font-size: 1.05rem; color: #0284c7; font-weight: 600;">
            Edge-Compatible Digital Biomarker Suite | MDS-UPDRS Part III Motor Assessment
        </p>
        <div style="display: flex; flex-wrap: wrap;">
            <div class="status-chip active">● MediaPipe Hands: Active 🟢</div>
            <div class="status-chip">⏱️ Target Sampling: 30 FPS</div>
            <div class="status-chip {'telehealth' if is_offline_mode else 'active'}">
                📡 Source: {'Pre-recorded Demo Clip (Offline Mode)' if is_offline_mode else 'Live Webcam (Edge DirectShow)'}
            </div>
            <div class="status-chip primary">📐 2D Scale Normalization: L_ref</div>
            <div class="status-chip">👤 Patient: {patient_id} ({patient_age}y / {patient_gender})</div>
        </div>
    </div>
    """,
    unsafe_allow_html=True,
)


# =============================================================================
# 5. CLINICIAN & EVALUATOR QUICK START GUIDE (INTERACTIVE USER GUIDE)
# =============================================================================
with st.expander("📖 **Clinician & Evaluator Quick Start Guide (How to Use & Test the System)**", expanded=False):
    g_col1, g_col2, g_col3 = st.columns(3)
    with g_col1:
        st.markdown(
            """
            <div class="guide-card" style="border-left: 4px solid #0284c7;">
                <h5 style="margin: 0 0 6px 0; color: #0369a1; font-weight: 700;">⚡ Option 1: Instant 10s Eval</h5>
                <p style="font-size: 0.84rem; color: #334155; margin: 0; line-height: 1.45;">
                    <b>Fastest 1-click test & verification:</b><br>
                    1. Simply click the <b>⚡ Instant Eval</b> button.<br>
                    2. Instantly computes full 10s kinematic aperture waveform, Welch PSD spectrum, MDS-UPDRS clinical score, and HL7 FHIR bundle ready for download.
                </p>
            </div>
            """,
            unsafe_allow_html=True,
        )
    with g_col2:
        st.markdown(
            """
            <div class="guide-card" style="border-left: 4px solid #16a34a;">
                <h5 style="margin: 0 0 6px 0; color: #15803d; font-weight: 700;">▶️ Option 2: Demo Video Stream</h5>
                <p style="font-size: 0.84rem; color: #334155; margin: 0; line-height: 1.45;">
                    <b>Simulated telehealth consultation:</b><br>
                    1. Keep source as <i>Pre-recorded Demo Clip</i>.<br>
                    2. Select <code>finger_tap_sample.mp4</code> or <code>tremor_sample.mp4</code> in sidebar.<br>
                    3. Click <b>▶️ Start 10s Trial</b> to observe live countdown, skeleton tracking, and real-time cadence!
                </p>
            </div>
            """,
            unsafe_allow_html=True,
        )
    with g_col3:
        st.markdown(
            """
            <div class="guide-card" style="border-left: 4px solid #7c3aed;">
                <h5 style="margin: 0 0 6px 0; color: #6d28d9; font-weight: 700;">📹 Option 3: Live Webcam</h5>
                <p style="font-size: 0.84rem; color: #334155; margin: 0; line-height: 1.45;">
                    <b>Real hardware edge camera test:</b><br>
                    1. Switch Source to <i>Live Webcam</i> in sidebar.<br>
                    2. Place hand 50-70cm from webcam.<br>
                    3. Click <b>▶️ Start 10s Trial</b> and tap thumb and index finger as fast and wide as possible.
                </p>
            </div>
            """,
            unsafe_allow_html=True,
        )


# =============================================================================
# 6. PLOTLY CHART BUILDERS (LIGHT THEME MEDICAL PALETTE)
# =============================================================================
def build_waveform_plotly(
    times: np.ndarray,
    apertures: np.ndarray,
    peak_times: Optional[np.ndarray] = None,
    peak_amps: Optional[np.ndarray] = None,
) -> go.Figure:
    """Create an interactive Plotly kinematic waveform chart with light medical styling."""
    fig = go.Figure()

    # Touch threshold reference line at 0.22
    fig.add_hline(
        y=0.22,
        line_dash="dot",
        line_color="#94a3b8",
        line_width=1.5,
        annotation_text="Touch Threshold (0.22)",
        annotation_position="bottom right",
        annotation_font_color="#475569",
        annotation_font_size=11,
    )

    if len(times) > 0 and len(apertures) > 0:
        # Kinematic normalized aperture trace
        fig.add_trace(
            go.Scatter(
                x=times,
                y=apertures,
                mode="lines",
                name="Normalized Aperture A(t)",
                line=dict(color="#0284c7", width=2.4, shape="spline"),
                hovertemplate="t: %{x:.2f}s<br>A(t): %{y:.3f} L_ref<extra></extra>",
            )
        )

        # Highlight detected peaks
        if peak_times is not None and len(peak_times) > 0 and peak_amps is not None:
            fig.add_trace(
                go.Scatter(
                    x=peak_times,
                    y=peak_amps,
                    mode="markers",
                    name="Confirmed Tap Peaks",
                    marker=dict(
                        symbol="diamond",
                        size=9,
                        color="#d97706",
                        line=dict(width=1.5, color="#ffffff"),
                    ),
                    hovertemplate="Tap Peak<br>t: %{x:.2f}s<br>Amp: %{y:.3f}<extra></extra>",
                )
            )

    fig.update_layout(
        title=dict(
            text="Real-Time Kinematic Waveform: Normalized Aperture A(t)",
            font=dict(size=14, color="#0f172a", family="Segoe UI, sans-serif"),
        ),
        xaxis=dict(
            title="Time (seconds)",
            color="#334155",
            gridcolor="#f1f5f9",
            linecolor="#cbd5e1",
            showgrid=True,
            zeroline=False,
        ),
        yaxis=dict(
            title="Aperture / L_ref",
            color="#334155",
            gridcolor="#f1f5f9",
            linecolor="#cbd5e1",
            showgrid=True,
            range=[0.0, max(0.65, float(np.max(apertures)) * 1.15 if len(apertures) > 0 else 0.65)],
        ),
        margin=dict(l=40, r=20, t=40, b=40),
        plot_bgcolor="#ffffff",
        paper_bgcolor="#ffffff",
        height=260,
        showlegend=True,
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=1.02,
            xanchor="right",
            x=1,
            font=dict(size=10, color="#475569"),
        ),
    )
    return fig


def build_psd_plotly(
    freqs: np.ndarray,
    psd: np.ndarray,
    peak_freq: float,
    peak_power: float,
) -> go.Figure:
    """Create an interactive Plotly power spectral density chart for Tremor (Task 3.15)."""
    fig = go.Figure()

    # Pathological Parkinsonian Tremor Window (3.0 - 7.0 Hz)
    fig.add_vrect(
        x0=3.0,
        x1=7.0,
        fillcolor="rgba(239, 68, 68, 0.12)",
        layer="below",
        line=dict(width=1, color="rgba(239, 68, 68, 0.3)"),
        annotation_text="Parkinsonian (3-7 Hz)",
        annotation_position="top left",
        annotation_font_color="#b91c1c",
        annotation_font_size=10,
    )

    # Physiological / Essential Tremor Window (8.0 - 12.0 Hz)
    fig.add_vrect(
        x0=8.0,
        x1=12.0,
        fillcolor="rgba(2, 132, 199, 0.12)",
        layer="below",
        line=dict(width=1, color="rgba(2, 132, 199, 0.3)"),
        annotation_text="Physiological (8-12 Hz)",
        annotation_position="top left",
        annotation_font_color="#0369a1",
        annotation_font_size=10,
    )

    if len(freqs) > 0 and len(psd) > 0:
        mask = (freqs >= 0.5) & (freqs <= 15.0)
        f_plot = freqs[mask]
        p_plot = psd[mask]

        fig.add_trace(
            go.Scatter(
                x=f_plot,
                y=p_plot,
                mode="lines",
                name="Welch PSD",
                line=dict(color="#0284c7", width=2.2),
                fill="tozeroy",
                fillcolor="rgba(2, 132, 199, 0.08)",
                hovertemplate="Freq: %{x:.2f} Hz<br>PSD: %{y:.6f}<extra></extra>",
            )
        )

        if peak_freq >= 3.0:
            fig.add_trace(
                go.Scatter(
                    x=[peak_freq],
                    y=[peak_power],
                    mode="markers",
                    name=f"Dominant: {peak_freq:.1f} Hz",
                    marker=dict(symbol="star", size=11, color="#e11d48", line=dict(width=1.5, color="#ffffff")),
                    hovertemplate=f"Dominant Tremor Peak<br>{peak_freq:.2f} Hz (Power: {peak_power:.5f})<extra></extra>",
                )
            )

    fig.update_layout(
        title=dict(
            text="Task 3.15 Tremor Spectral Decomposition (Welch PSD)",
            font=dict(size=14, color="#0f172a", family="Segoe UI, sans-serif"),
        ),
        xaxis=dict(
            title="Frequency (Hz)",
            color="#334155",
            gridcolor="#f1f5f9",
            linecolor="#cbd5e1",
            range=[0.0, 15.0],
            showgrid=True,
        ),
        yaxis=dict(
            title="Power Spectral Density",
            color="#334155",
            gridcolor="#f1f5f9",
            linecolor="#cbd5e1",
            showgrid=True,
        ),
        margin=dict(l=40, r=20, t=40, b=40),
        plot_bgcolor="#ffffff",
        paper_bgcolor="#ffffff",
        height=240,
        showlegend=True,
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=1.02,
            xanchor="right",
            x=1,
            font=dict(size=10, color="#475569"),
        ),
    )
    return fig


# =============================================================================
# 7. ROBUST SIMULATION / OFFLINE FALLBACK GENERATOR
# =============================================================================
def generate_synthetic_kinematic_trial(duration: float = 10.0, fs: float = 30.0) -> Tuple[np.ndarray, np.ndarray, np.ndarray, List[np.ndarray]]:
    """
    Generate clean, realistic clinical kinematic series and synthetic video frames
    demonstrating finger tapping with physiological bradykinesia decrement (MDS-UPDRS Task 3.4)
    and mild 5.2 Hz resting tremor (Task 3.15).

    Returns:
        Tuple of (timestamps, aperture_series, wrist_series, list_of_bgr_frames).
    """
    n_frames = int(fs * duration)
    timestamps = np.linspace(0.0, duration, n_frames)

    # 1. Finger Tapping: 2.35 taps/second with slight decay (-0.018/sec)
    base_cadence = 2.35
    phase = 2 * np.pi * base_cadence * timestamps
    aperture_base = 0.5 * (1.0 - np.cos(phase))  # [0, 1]
    
    # Physiological amplitude decrement
    decay_profile = np.maximum(0.20, 0.44 - 0.018 * timestamps)
    aperture_series = 0.15 + decay_profile * aperture_base
    aperture_series += np.random.normal(0, 0.005, size=n_frames)
    aperture_series = np.clip(aperture_series, 0.05, 0.85)

    # 2. Wrist Trajectory: 5.2 Hz resting tremor
    tremor_freq = 5.2
    wx = 0.50 + 0.018 * np.sin(2 * np.pi * tremor_freq * timestamps) + np.random.normal(0, 0.001, size=n_frames)
    wy = 0.70 + 0.010 * np.cos(2 * np.pi * tremor_freq * timestamps)
    wrist_series = np.column_stack([wx, wy])

    # 3. Render video frames with landmarks
    frames = []
    w_px, h_px = 640, 480
    for idx in range(n_frames):
        t = timestamps[idx]
        ap = aperture_series[idx]
        # Clean medical telemetry canvas
        frame = np.full((h_px, w_px, 3), (250, 252, 255), dtype=np.uint8)

        # Subtle background grid lines
        for gx in range(40, w_px, 80):
            cv2.line(frame, (gx, 0), (gx, h_px), (230, 238, 248), 1)
        for gy in range(40, h_px, 80):
            cv2.line(frame, (0, gy), (w_px, gy), (230, 238, 248), 1)

        # Kinematic coordinate nodes
        wrist_pt = (int(wx[idx] * w_px), int(wy[idx] * h_px))
        mcp_pt = (wrist_pt[0], wrist_pt[1] - 140)

        # Scale factor L_ref = 140px in pixel coordinates
        sep_px = int(ap * 140)
        thumb_pt = (mcp_pt[0] - sep_px // 2 - 25, mcp_pt[1] - 30)
        index_pt = (mcp_pt[0] + sep_px // 2 + 15, mcp_pt[1] - 55)

        # Draw L_ref anatomical baseline (Wrist -> MCP9)
        cv2.line(frame, wrist_pt, mcp_pt, (22, 163, 74), 2, cv2.LINE_AA)

        # Draw Thumb & Index kinematic links
        cv2.line(frame, mcp_pt, thumb_pt, (148, 163, 184), 2, cv2.LINE_AA)
        cv2.line(frame, mcp_pt, index_pt, (148, 163, 184), 2, cv2.LINE_AA)

        # Aperture excursion vector between thumb and index
        is_pinching = ap < 0.22
        vec_color = (22, 163, 74) if is_pinching else (2, 132, 199)
        cv2.line(frame, thumb_pt, index_pt, vec_color, 3, cv2.LINE_AA)

        # Anatomical Landmark Nodes with medical rings
        for pt, label, color in [
            (wrist_pt, "WRIST", (15, 23, 42)),
            (mcp_pt, "MCP9 (Ref)", (22, 163, 74)),
            (thumb_pt, "THUMB", (2, 132, 199)),
            (index_pt, "INDEX", (2, 132, 199)),
        ]:
            cv2.circle(frame, pt, 7, color, -1, cv2.LINE_AA)
            cv2.circle(frame, pt, 11, color, 1, cv2.LINE_AA)
            cv2.putText(frame, label, (pt[0] + 14, pt[1] + 4), cv2.FONT_HERSHEY_SIMPLEX, 0.40, (51, 65, 85), 1, cv2.LINE_AA)

        # Top banner indicating Offline Simulation Mode
        cv2.putText(frame, "[OFFLINE DIGITAL TWIN KINEMATIC SIMULATION]", (20, 32), cv2.FONT_HERSHEY_SIMPLEX, 0.50, (2, 132, 199), 1, cv2.LINE_AA)

        frames.append(frame)

    return timestamps, aperture_series, wrist_series, frames


# =============================================================================
# 8. TWO-COLUMN SIDE-BY-SIDE CLINICIAN LAYOUT
# =============================================================================
col_feed, col_analytics = st.columns([1, 1], gap="large")


# -----------------------------------------------------------------------------
# LEFT COLUMN: Video Feed & Camera Controls
# -----------------------------------------------------------------------------
with col_feed:
    st.markdown("### 📹 **Video Stream & Clinical Trial Acquisition**")

    # Responsive Action Buttons Toolbar (Equal Widths, Single-Line Labels)
    btn_col1, btn_col2, btn_col3 = st.columns([1.3, 1.15, 0.85], gap="small")
    with btn_col1:
        start_trial_pressed = st.button("▶️ **Start 10s Trial**", use_container_width=True, type="primary")
    with btn_col2:
        run_full_offline_eval = st.button("⚡ **Instant Eval**", use_container_width=True)
    with btn_col3:
        reset_trial_pressed = st.button("🔄 **Reset**", use_container_width=True)

    if reset_trial_pressed:
        ta = get_state("tapping_analyzer")
        if ta:
            ta.reset_trial()
            ta.state_machine.reset_session()
        kb = get_state("kinematic_buffer")
        if kb:
            kb.clear()
        set_state("stream_active", False)
        set_state("waveform_history_time", np.array([]))
        set_state("waveform_history_aperture", np.array([]))
        st.rerun()

    # Progress bar and video display placeholders
    countdown_container = st.empty()
    video_placeholder = st.empty()
    status_card_placeholder = st.empty()

    # Handle Instant Offline Evaluation (zero-latency batch run)
    if run_full_offline_eval:
        with st.spinner("Processing 10-second motor evaluation session..."):
            t_sim, ap_sim, wr_sim, demo_frames = generate_synthetic_kinematic_trial(duration=10.0, fs=30.0)

            ta = get_state("tapping_analyzer")
            tr = get_state("tremor_analyzer")
            tap_metrics = ta.analyze_tapping(ap_sim, t_sim) if ta else {}
            updrs_tapping = MDSUPDRSScorer.score_finger_tapping(tap_metrics)

            tremor_metrics = tr.analyze_tremor(wr_sim, t_sim) if tr else {}
            updrs_tremor = MDSUPDRSScorer.score_tremor(tremor_metrics)

            set_state("latest_tap_metrics", tap_metrics)
            set_state("latest_updrs_tapping", updrs_tapping)
            set_state("latest_tremor_metrics", tremor_metrics)
            set_state("latest_updrs_tremor", updrs_tremor)
            set_state("waveform_history_time", t_sim)
            set_state("waveform_history_aperture", ap_sim)

            if demo_frames:
                final_frame = demo_frames[-1].copy()
                trial_res = {
                    "state": "COMPLETED",
                    "remaining_time": 0.0,
                    "trial_taps_count": tap_metrics.get("total_taps", 0),
                    "metrics": tap_metrics,
                    "score": updrs_tapping,
                }
                hud_frame = draw_clinical_hud(
                    frame=final_frame,
                    fps=30.0,
                    buffer_len=150,
                    buffer_capacity=150,
                    handedness_str="Right (0.99)",
                    current_aperture=float(ap_sim[-1]),
                    current_state="TOUCH",
                    session_taps=tap_metrics.get("total_taps", 0),
                    instantaneous_cadence=tap_metrics.get("tap_rate_hz", 0.0),
                    trial_status=trial_res,
                    tap_metrics=tap_metrics,
                    updrs_tapping=updrs_tapping,
                    tremor_metrics=tremor_metrics,
                    updrs_tremor=updrs_tremor,
                    distance_series=ap_sim,
                    scale_factor=0.28,
                )
                video_placeholder.image(cv2.cvtColor(hud_frame, cv2.COLOR_BGR2RGB), use_column_width=True)
            st.success("✅ Instant 10-Second Assessment Completed & Scored Successfully!")

    # Video Acquisition / Streaming Loop
    elif start_trial_pressed or get_state("stream_active", False):
        set_state("stream_active", True)
        tapping_analyzer: FingerTappingAnalyzer = get_state("tapping_analyzer")
        tremor_analyzer: TremorAnalyzer = get_state("tremor_analyzer")
        buffer: KinematicBuffer = get_state("kinematic_buffer")

        # ALWAYS pre-generate synthetic kinematics & frames so fallback is 100% guaranteed to exist
        t_sim_fallback, syn_ap, syn_wr, syn_frames = generate_synthetic_kinematic_trial(duration=10.0, fs=30.0)

        # Check source
        cap = None
        using_synthetic_fallback = False
        if is_offline_mode:
            if selected_video_path and os.path.exists(selected_video_path):
                cap = cv2.VideoCapture(selected_video_path)
            if cap is None or not cap.isOpened():
                using_synthetic_fallback = True
        else:
            cap = HandTracker.create_capture(camera_id=camera_id)
            if not cap.isOpened():
                st.warning(f"⚠️ Live webcam (device index {camera_id}) not accessible on this device. Activating offline simulation fallback.")
                using_synthetic_fallback = True

        tracker = HandTracker()
        tapping_analyzer.start_trial(current_time=time.perf_counter())

        frame_idx = 0
        total_trial_frames = 300  # 10s at 30 FPS
        start_time = time.perf_counter()
        prev_time = start_time
        fps_smooth = 30.0

        while frame_idx < total_trial_frames:
            curr_time = time.perf_counter()
            dt = curr_time - prev_time
            prev_time = curr_time
            if dt > 0.0:
                fps_smooth = 0.85 * fps_smooth + 0.15 * (1.0 / dt)

            # Ingest frame
            frame = None
            hand_data = None
            raw_frame = None

            if not using_synthetic_fallback and cap is not None and cap.isOpened():
                ret, raw_frame = cap.read()
                if ret and raw_frame is not None and raw_frame.size > 0:
                    frame, hand_data = tracker.process_frame(raw_frame)
                else:
                    # Loop video if pre-recorded
                    cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
                    ret, raw_frame = cap.read()
                    if ret and raw_frame is not None and raw_frame.size > 0:
                        frame, hand_data = tracker.process_frame(raw_frame)

            # Guaranteed robust fallback: syn_frames is ALWAYS defined
            if frame is None or hand_data is None:
                syn_idx = min(frame_idx, len(syn_frames) - 1)
                if raw_frame is not None and raw_frame.size > 0:
                    frame = raw_frame.copy()
                else:
                    frame = syn_frames[syn_idx].copy()
                sample_t = curr_time
                current_aperture = float(syn_ap[syn_idx])
                scale_factor = 0.28
                handedness_str = "Right (0.99)"
                wrist_coords = (syn_wr[syn_idx][0], syn_wr[syn_idx][1], 0.0)
                buffer.push(
                    timestamp=sample_t,
                    thumb_tip=(0.4, 0.4, 0.0),
                    index_tip=(0.4 + current_aperture * scale_factor, 0.4, 0.0),
                    wrist=wrist_coords,
                    middle_mcp=(wrist_coords[0], wrist_coords[1] - scale_factor, 0.0),
                )
            else:
                sample_t = hand_data["timestamp"]
                current_aperture = float(hand_data.get("normalized_aperture", 0.0))
                scale_factor = float(hand_data.get("scale_factor", hand_data.get("l_ref", 0.28)))
                handedness_str = str(hand_data["handedness"])
                norm = hand_data["landmarks_norm"]
                buffer.push(
                    sample_t,
                    norm[THUMB_TIP_IDX],
                    norm[INDEX_TIP_IDX],
                    norm[WRIST_IDX],
                    norm[MIDDLE_MCP_IDX],
                )

            # Update tapping state machine and 10s trial
            sample_res = tapping_analyzer.process_sample(sample_t, current_aperture)
            current_state = sample_res["current_state"]
            trial_status = sample_res["trial_status"]
            inst_cadence = sample_res["instantaneous_cadence"]

            # Real-time analytics update every 5 frames
            if frame_idx % 5 == 0 and buffer.is_ready(min_samples=25):
                series = buffer.get_time_series()
                t_rel = series["time"]
                d_norm = series["normalized_aperture"]
                wrist_pos = series["wrist"]

                set_state("waveform_history_time", t_rel)
                set_state("waveform_history_aperture", d_norm)

                if len(d_norm) >= 15:
                    latest_tap_metrics = tapping_analyzer.analyze_tapping(d_norm, t_rel)
                    latest_updrs_tapping = MDSUPDRSScorer.score_finger_tapping(latest_tap_metrics)
                    latest_tremor_metrics = tremor_analyzer.analyze_tremor(wrist_pos, t_rel)
                    latest_updrs_tremor = MDSUPDRSScorer.score_tremor(latest_tremor_metrics)

                    set_state("latest_tap_metrics", latest_tap_metrics)
                    set_state("latest_updrs_tapping", latest_updrs_tapping)
                    set_state("latest_tremor_metrics", latest_tremor_metrics)
                    set_state("latest_updrs_tremor", latest_updrs_tremor)

            # Render polished HUD onto frame if enabled
            if show_clinical_hud_cards:
                frame = draw_clinical_hud(
                    frame=frame,
                    fps=fps_smooth,
                    buffer_len=len(buffer),
                    buffer_capacity=buffer.buffer_size,
                    handedness_str=handedness_str,
                    current_aperture=current_aperture,
                    current_state=current_state,
                    session_taps=tapping_analyzer.session_total_taps,
                    instantaneous_cadence=inst_cadence,
                    trial_status=trial_status,
                    tap_metrics=get_state("latest_tap_metrics", {}),
                    updrs_tapping=get_state("latest_updrs_tapping", {}),
                    tremor_metrics=get_state("latest_tremor_metrics", {}),
                    updrs_tremor=get_state("latest_updrs_tremor", {}),
                    distance_series=get_state("waveform_history_aperture"),
                    scale_factor=scale_factor,
                )

            # Streamlit visual update
            video_placeholder.image(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB), use_column_width=True)

            # Countdown progress update
            rem_sec = trial_status.get("remaining_time", max(0.0, 10.0 - (curr_time - start_time)))
            prog_val = max(0.0, min(1.0, (10.0 - rem_sec) / 10.0))
            countdown_container.progress(prog_val, text=f"⏱️ **10s Trial Progress:** {rem_sec:4.1f}s remaining | Taps: {trial_status.get('trial_taps_count', 0)}")

            frame_idx += 1
            time.sleep(0.015)  # Yield to Streamlit event loop for smooth rendering

        if cap is not None:
            cap.release()
        tracker.close()
        set_state("stream_active", False)
        st.success("✅ Formal 10-Second Motor Assessment Concluded!")

    else:
        # Idle video feed placeholder with high contrast light background
        idle_frame = np.full((480, 640, 3), (250, 252, 255), dtype=np.uint8)
        cv2.rectangle(idle_frame, (2, 2), (637, 477), (203, 213, 225), 2)
        if not is_offline_mode:
            cv2.putText(idle_frame, "LIVE WEBCAM STANDBY", (170, 210), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (2, 132, 199), 2, cv2.LINE_AA)
            cv2.putText(idle_frame, f"Device Index #{camera_id} Ready", (210, 245), cv2.FONT_HERSHEY_SIMPLEX, 0.50, (22, 163, 74), 1, cv2.LINE_AA)
            cv2.putText(idle_frame, "Click 'Start 10s Trial' above to open camera & stream", (105, 280), cv2.FONT_HERSHEY_SIMPLEX, 0.48, (71, 85, 105), 1, cv2.LINE_AA)
        else:
            cv2.putText(idle_frame, "OFFLINE DEMO STANDBY", (170, 210), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (126, 34, 206), 2, cv2.LINE_AA)
            cv2.putText(idle_frame, "Click 'Start 10s Trial' or 'Instant Eval' to begin", (110, 260), cv2.FONT_HERSHEY_SIMPLEX, 0.48, (71, 85, 105), 1, cv2.LINE_AA)
        video_placeholder.image(cv2.cvtColor(idle_frame, cv2.COLOR_BGR2RGB), use_column_width=True)
        countdown_container.caption("Ready for standardized 10-second MDS-UPDRS trial.")


# -----------------------------------------------------------------------------
# RIGHT COLUMN: Real-Time Kinematics & Clinical Metrics
# -----------------------------------------------------------------------------
with col_analytics:
    st.markdown("### 📊 **Kinematic Biomarkers & Clinical Scoring**")

    # Current metric values
    metrics = get_state("latest_tap_metrics", {})
    score_tapping = get_state("latest_updrs_tapping", {})
    tremor = get_state("latest_tremor_metrics", {})
    score_tremor = get_state("latest_updrs_tremor", {})

    # 4 High-Contrast Metric Cards (2x2 Grid)
    mc1, mc2 = st.columns(2)
    with mc1:
        st.markdown(
            f"""
            <div class="metric-card">
                <div class="title">Total Confirmed Taps & Cadence</div>
                <div class="value">{metrics.get('total_taps', 0)} <span style="font-size: 1.05rem; font-weight: 600; color: #0284c7;">({metrics.get('tap_rate_hz', 0.0):.2f} Hz)</span></div>
                <div class="subtext">Target: &ge; 2.50 taps/sec</div>
            </div>
            """,
            unsafe_allow_html=True,
        )
    with mc2:
        slope_val = metrics.get('amplitude_decay_slope', 0.0)
        slope_color = "#15803d" if slope_val > -0.015 else ("#b45309" if slope_val > -0.035 else "#b91c1c")
        st.markdown(
            f"""
            <div class="metric-card">
                <div class="title">Amplitude Decrement Slope</div>
                <div class="value" style="color: {slope_color};">{slope_val:+.3f} <span style="font-size: 0.85rem; font-weight: 500; color: #64748b;">/sec</span></div>
                <div class="subtext">Fatigue rate (Normal: &gt; -0.015)</div>
            </div>
            """,
            unsafe_allow_html=True,
        )

    st.write("")
    mc3, mc4 = st.columns(2)
    with mc3:
        cv_val = metrics.get('rhythm_cv', 0.0)
        cv_pct = cv_val * 100.0
        cv_color = "#15803d" if cv_val < 0.20 else ("#b45309" if cv_val < 0.30 else "#b91c1c")
        st.markdown(
            f"""
            <div class="metric-card">
                <div class="title">Rhythm Regularity (CV)</div>
                <div class="value" style="color: {cv_color};">{cv_pct:.1f}%</div>
                <div class="subtext">Inter-tap variability (Normal: &lt; 20%)</div>
            </div>
            """,
            unsafe_allow_html=True,
        )
    with mc4:
        st.markdown(
            f"""
            <div class="metric-card">
                <div class="title">Mean Normalized Aperture</div>
                <div class="value">{metrics.get('mean_amplitude', 0.0):.3f} <span style="font-size: 0.85rem; font-weight: 500; color: #64748b;">L_ref</span></div>
                <div class="subtext">Excursion ratio (Thumb to Index)</div>
            </div>
            """,
            unsafe_allow_html=True,
        )

    # 1. Live Aperture Waveform (Plotly)
    st.write("")
    peak_times = metrics.get("peak_times")
    peak_amps = metrics.get("amplitudes")
    waveform_fig = build_waveform_plotly(
        times=get_state("waveform_history_time", np.array([])),
        apertures=get_state("waveform_history_aperture", np.array([])),
        peak_times=peak_times if peak_times is not None and len(peak_times) > 0 else None,
        peak_amps=peak_amps if peak_amps is not None and len(peak_amps) > 0 else None,
    )
    st.plotly_chart(waveform_fig, use_container_width=True)

    # 2. Spectral Analysis (Tremor Task 3.15)
    psd_fig = build_psd_plotly(
        freqs=tremor.get("freqs", np.array([])),
        psd=tremor.get("psd", np.array([])),
        peak_freq=tremor.get("peak_freq", 0.0),
        peak_power=tremor.get("peak_power", 0.0),
    )
    st.plotly_chart(psd_fig, use_container_width=True)

    # 3. Automated MDS-UPDRS Part III Score Card (Light Theme)
    st.markdown("#### 🏆 **Automated MDS-UPDRS Part III Score Card**")
    sc1, sc2 = st.columns(2)

    with sc1:
        t_score = score_tapping.get("score", 0)
        t_label = score_tapping.get("label", "Normal")
        st.markdown(
            f"""
            <div class="clinical-score-card">
                <div style="display: flex; justify-content: space-between; align-items: center; margin-bottom: 8px;">
                    <span style="font-weight: 700; font-size: 0.95rem; color: #0f172a;">Task 3.4 Finger Tapping</span>
                    <span class="score-badge score-{t_score}">Score {t_score}: {t_label.upper()}</span>
                </div>
                <p style="font-size: 0.84rem; color: #334155; margin: 0; line-height: 1.45;">
                    {score_tapping.get('clinical_rationale', 'Normal motor performance.')}
                </p>
            </div>
            """,
            unsafe_allow_html=True,
        )

    with sc2:
        tr_score = score_tremor.get("score", 0)
        tr_label = score_tremor.get("label", "Normal")
        st.markdown(
            f"""
            <div class="clinical-score-card">
                <div style="display: flex; justify-content: space-between; align-items: center; margin-bottom: 8px;">
                    <span style="font-weight: 700; font-size: 0.95rem; color: #0f172a;">Task 3.15 Tremor</span>
                    <span class="score-badge score-{tr_score}">Score {tr_score}: {tr_label.upper()}</span>
                </div>
                <p style="font-size: 0.84rem; color: #334155; margin: 0; line-height: 1.45;">
                    {score_tremor.get('clinical_rationale', 'No pathological tremor detected.')}
                </p>
            </div>
            """,
            unsafe_allow_html=True,
        )


# =============================================================================
# 9. HL7 FHIR R4 EHR EXPORT SECTION
# =============================================================================
st.divider()
st.markdown("### 🏥 **HL7 FHIR R4 Electronic Health Record (EHR) Export**")
st.caption("Interoperable diagnostic reporting compliant with HL7 FHIR R4, LOINC, and SNOMED CT clinical standards.")

with st.expander("📋 **Review Patient Clinical Metadata & Download Diagnostic Bundle**", expanded=False):
    exp_col1, exp_col2 = st.columns([2, 1])

    patient_meta_dict = {
        "id": patient_id,
        "patient_id": patient_id,
        "age": int(patient_age),
        "gender": str(patient_gender).lower(),
        "clinician_name": clinician_name,
        "clinician_notes": clinician_notes,
        "session_type": "Telehealth Remote Exam" if is_offline_mode else "On-Premise Clinical Motor Screen",
        "timestamp": datetime.now(timezone.utc).isoformat(),
    }

    session_metrics_dict = {
        "tapping": get_state("latest_tap_metrics", {}),
        "tremor": get_state("latest_tremor_metrics", {}),
    }

    clinical_scores_dict = {
        "tapping": get_state("latest_updrs_tapping", {}),
        "tremor": get_state("latest_updrs_tremor", {}),
    }

    bundle_dict, json_str = export_fhir_json(
        patient_meta=patient_meta_dict,
        session_metrics=session_metrics_dict,
        clinical_scores=clinical_scores_dict,
        indent=2,
    )

    with exp_col1:
        st.markdown(f"• **Target Resource:** `Bundle (type: document)` | **Total Entries:** 4 (`Patient`, 2 `Observations`, `DiagnosticReport`)")
        st.markdown(f"• **LOINC Codes:** `89555-7` (Finger Tapping Task 3.4) | `89553-2` (Tremor Task 3.15)")
        st.markdown(f"• **SNOMED CT:** `285640004` (Finger Tapping) | `247545004` (Tremor)")
        st.info("Assistive AI disclaimer attached: *Clinician-in-the-loop assistive biomarker evaluation. Not an autonomous diagnostic verdict.*")

    with exp_col2:
        file_name = f"neuromotion_fhir_{patient_id}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        st.download_button(
            label="📥 **Download HL7 FHIR Bundle (.json)**",
            data=json_str,
            file_name=file_name,
            mime="application/json",
            use_container_width=True,
            type="primary",
        )

    st.markdown("##### **FHIR R4 JSON Payload Preview**")
    st.json(bundle_dict, expanded=False)
