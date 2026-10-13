"""
NeuroMotion - Edge Computer Vision & Digital Biomarker System
=============================================================
Real-time motor and tremor kinematic analysis (MDS-UPDRS).
"""

import streamlit as st

st.set_page_config(
    page_title="NeuroMotion AI | Motor & Tremor Biomarkers",
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Sidebar Configuration
with st.sidebar:
    st.image("https://img.icons8.com/fluency/96/artificial-intelligence.png", width=64)
    st.title("NeuroMotion Control")
    st.caption("Edge-First Digital Biomarker Platform")
    st.divider()

    st.subheader("Assessment Mode")
    assessment_mode = st.selectbox(
        "Select Clinical Protocol",
        [
            "MDS-UPDRS 3.17: Rest Tremor",
            "MDS-UPDRS 3.15: Postural Tremor",
            "MDS-UPDRS 3.4: Finger Tapping",
            "MDS-UPDRS 3.5: Hand Movements",
            "Full Motor Screen",
        ],
    )

    st.subheader("Input Stream")
    input_source = st.radio("Source", ["Live Onboard Camera (Edge)", "Upload Test Video"])

    st.divider()
    st.markdown("**System Status:** `Ready` 🟢")
    st.caption("Target Sampling: 30 FPS | Pipeline: MediaPipe + Butterworth")

# Main Dashboard View
st.title("🧠 NeuroMotion AI")
st.subheader("Edge-First Motor & Tremor Digital Biomarker Analysis (MDS-UPDRS)")

col1, col2, col3, col4 = st.columns(4)
with col1:
    st.metric(label="Patient ID", value="PT-1042")
with col2:
    st.metric(label="Dominant Frequency", value="-- Hz", delta="Target: 4-6 Hz")
with col3:
    st.metric(label="Tremor Amplitude", value="-- cm", delta="MDS: Stage 0")
with col4:
    st.metric(label="Inference Latency", value="-- ms", delta="Edge Optimized")

st.divider()

col_feed, col_analytics = st.columns([3, 2])

with col_feed:
    st.info("📹 **Live Video Feed**: Camera feed and landmark overlay will initialize here.")
    st.container(height=380, border=True).write("Awaiting stream acquisition...")

with col_analytics:
    st.info("📊 **Real-Time Kinematics & Spectral Power**")
    st.container(height=380, border=True).write("Kinematic signal processing graphs will render here.")
