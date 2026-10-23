"""
NeuroMotion Core Module
=======================
Vision processing, real-time landmark tracking, signal filtering,
and kinematic digital biomarker extraction.
"""

from .vision import (
    HandTracker,
    KinematicBuffer,
    VisionPipeline,
    WRIST_IDX,
    THUMB_TIP_IDX,
    INDEX_TIP_IDX,
    MIDDLE_MCP_IDX,
    MIDDLE_TIP_IDX,
    KEY_LANDMARK_INDICES,
    draw_clinical_hud,
    draw_trial_complete_overlay,
)
from .signal_processing import SignalFilter, SignalProcessor
from .biomarkers import (
    BiomarkerExtractor,
    FingerTappingAnalyzer,
    TremorAnalyzer,
    TapState,
    TapEvent,
    TapStateMachine,
    TappingTrialManager,
)

__all__ = [
    "HandTracker",
    "KinematicBuffer",
    "VisionPipeline",
    "WRIST_IDX",
    "THUMB_TIP_IDX",
    "INDEX_TIP_IDX",
    "MIDDLE_MCP_IDX",
    "MIDDLE_TIP_IDX",
    "KEY_LANDMARK_INDICES",
    "draw_clinical_hud",
    "draw_trial_complete_overlay",
    "SignalFilter",
    "SignalProcessor",
    "BiomarkerExtractor",
    "FingerTappingAnalyzer",
    "TremorAnalyzer",
    "TapState",
    "TapEvent",
    "TapStateMachine",
    "TappingTrialManager",
]
