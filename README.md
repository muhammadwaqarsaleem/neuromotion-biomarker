# NeuroMotion: Edge-Compatible Markerless Video Biomarker Engine for Motor and Tremor Assessment

## Project Overview & Problem Statement
Neurological and movement disorders—including Parkinson’s disease, essential tremor, ataxia, and post-stroke motor impairment—affect tens of millions globally. Standard clinical evaluation relies on qualitative rating scales (e.g., MDS-UPDRS, Unified Parkinson's Disease Rating Scale) conducted in-clinic by scarce movement disorder specialists. 

This creates three critical bottlenecks:
1. **Severe Diagnostic Delays & Access Gaps:** Patients face months-long waitlists to see specialized neurologists.
2. **Subjective & Episodic Assessment:** Clinical examinations provide only a snapshot in time, missing episodic fluctuations and suffering from inter-rater variability.
3. **Prohibitive Hardware Costs:** Dedicated laboratory gait-and-motion capture systems (e.g., Vicon, wearable sensor arrays) cost tens of thousands of dollars, confining objective tracking to elite tertiary hospitals.

## Proposed Solution
**NeuroMotion** is an end-to-end, markerless computer-vision platform that converts any standard consumer video feed (RGB webcam or smartphone) into quantitative, clinically validated motor biomarkers. 

By extracting sub-millimeter 3D spatial joint coordinates in real time and feeding temporal kinematic time-series into lightweight deep neural models, NeuroMotion quantifies tremor frequency, postural sway, finger-tapping bradykinesia, and gait asymmetry at clinical fidelity without requiring wearables, markers, or specialized sensor hardware.

## Objectives & Expected Outcomes
* **Objective 1:** Build a high-throughput video pipeline extracting 33 full-body and 21 hand skeletal keypoints at $\ge 30\text{ FPS}$ on standard consumer hardware.
* **Objective 2:** Implement a temporal sequence model to quantify motor symptom metrics (tremor frequency between $3\text{--}12\text{ Hz}$, tap frequency, amplitude decay, and movement freezing).
* **Objective 3:** Map algorithmic scores to standard MDS-UPDRS Part III clinical scores with $\ge 85\%$ clinician correlation ($R^2 \ge 0.82$).
* **Objective 4:** Provide a real-time, deterministic web interface that generates an interpretable, FHIR-compliant diagnostic summary for healthcare providers.

## Target Users
* **Neurologists & Movement Disorder Specialists:** For objective, standardized pre-consultation screenings and longitudinal treatment tracking.
* **Primary Care Physicians & Rural Clinics:** As an automated triage tool to identify early-stage motor symptoms before referring to tertiary hospitals.
* **Clinical Trial Organizations & Researchers:** For continuous, objective drug-response monitoring in neurodegenerative therapy trials.
* **Patients & Caregivers:** For frictionless, at-home remote motor rehabilitation tracking via telemedicine.

## Key Features / Deliverables
1. **Markerless 3D Keypoint Extraction Pipeline:** Real-time pose and hand topological tracking utilizing optimized lightweight vision backbones (MediaPipe / RTMPose).
2. **Kinematic Signal Processing & Tremor Spectral Analysis:** Fast Fourier Transform (FFT) and Wavelet Decomposition extracting tremor amplitude, rhythmicity, and peak frequencies.
3. **Temporal Motor Anomaly Classifier:** A Temporal Convolutional Network (TCN) trained on kinematic trajectories to grade bradykinesia, hypokinesia, and dyskinesia severity.
4. **Clinician Explainability & Saliency Overlay:** Visual kinematic trajectory overlays, kinematic velocity phase plots, and confidence calibration bounds rather than black-box outputs.
5. **Interoperable Electronic Health Record (EHR) Export:** Automated generation of Fast Healthcare Interoperability Resources (FHIR) JSON reports detailing longitudinal motor scores.

## Implementation Plan & Timeline
* **Days 1–2 (Oct 13–14): Core Landmark Extraction & Signal Ingestion**
  * Integrate real-time webcam video stream with MediaPipe Hands/Pose backbones.
  * Implement time-series buffering for 3D coordinates (wrist, index finger, thumb) with fixed temporal sampling (30 FPS).
  * Apply Savitzky-Golay filtering and cubic-spline interpolation for coordinate smoothing and jitter reduction.

* **Days 3–4 (Oct 15–16): Kinematic Signal Processing & Biomarker Engine**
  * Implement Fast Fourier Transform (FFT) and Welch’s method to extract peak spectral tremor frequencies ($3\text{--}12\text{ Hz}$).
  * Formulate kinematic algorithms for finger-tapping motor metrics: tap frequency, amplitude decrement (fatigue), and inter-tap interval variance.
  * Map extracted kinematic metrics to standardized MDS-UPDRS Part III clinical severity tiers (Normal: 0 to Severe: 4).

* **Days 5–7 (Oct 17–19): Interactive Dashboard & Real-Time Visualization**
  * Build the clinician evaluation dashboard (Streamlit / FastAPI + React interface).
  * Integrate live synchronized telemetry: webcam skeleton overlay alongside real-time Matplotlib/Plotly kinematic phase plots and spectral density graphs.
  * Implement one-click FHIR-compliant JSON/PDF diagnostic summary generation.

* **Days 8–9 (Oct 20–21): System Benchmarking, Latency Optimization & Edge Packaging**
  * Run end-to-end latency benchmarks on local CPU to guarantee deterministic, offline $\ge 30\text{ FPS}$ execution without cloud API dependencies.
  * Stress-test edge cases: camera lighting variations, partial hand occlusions, and recovery from dropped frames.

* **Day 10 (Oct 22–23): Live Pitch Preparation & Stage Rehearsal**
  * Dry-run live volunteer testing to ensure flawless plug-and-play demonstration during the championship presentation on October 24.

## Resources & Requirements
* **Software Stack:** Python 3.10+, PyTorch, OpenCV, MediaPipe, ONNX Runtime, Next.js / TailwindCSS, FastAPI.
* **Datasets:** Open-source biomechanical and movement datasets, including Kaggle Parkinson's Disease Movement Data, PhysioNet gait recordings, and synthesized kinematic data.
* **Hardware Requirements:** A standard laptop with an integrated 720p/1080p webcam and an entry-level GPU (e.g., NVIDIA GTX 1650 or modern Apple Silicon) for training; inference runs locally on standard CPU/WebAssembly.

## Potential Challenges & Risks
* **Lighting, Occlusion, and Varying Camera Angles:** Mitigated through spatial normalization, affine coordinate transformations, and synthetic viewpoint data augmentation.
* **Frame Rate Fluctuations:** Low-end smartphone webcams dropping frames could introduce noise into FFT spectrums. Mitigated by applying cubic-spline interpolation and fixed-interval resampling before frequency analysis.
* **Clinical Trust and False Positives:** Addressed by surfacing conformal prediction confidence intervals and explicit "Low Confidence / Refer to Manual Review" warnings when tracking quality drops.

## Additional Details
NeuroMotion is built strictly following edge-first architectural principles. All video processing and coordinate extractions occur entirely client-side or on on-premise edge runtimes, ensuring patient biometric video data never leaves the local machine—eliminating HIPAA and GDPR patient privacy vulnerabilities.
