"""
MDS-UPDRS Scoring Rules Engine
==============================
Maps kinematic biomarkers (frequency, amplitude decrement, rhythm regularity,
tremor power) to standardized Movement Disorder Society - Unified Parkinson's
Disease Rating Scale (MDS-UPDRS) Part III motor assessment scores.

Rating Scale:
    0: Normal
    1: Slight
    2: Mild
    3: Moderate
    4: Severe
"""

from typing import Any, Dict, List, Optional


class MDSUPDRSScorer:
    """
    Evaluates motor performance based on MDS-UPDRS Part III clinical criteria:
        - Task 3.4: Finger Tapping (speed, amplitude decrement, rhythm regularity)
        - Tasks 3.15 - 3.18: Tremor Assessment (postural, kinetic, and resting tremor)
    """

    SCORE_LABELS: Dict[int, str] = {
        0: "Normal",
        1: "Slight",
        2: "Mild",
        3: "Moderate",
        4: "Severe",
    }

    @classmethod
    def score_finger_tapping(cls, tap_metrics: Dict[str, Any]) -> Dict[str, Any]:
        """
        Score MDS-UPDRS Part III (Task 3.4 - Finger Tapping).

        Evaluates tap rate (speed/bradykinesia), amplitude decay slope (decrement/fatigue),
        and coefficient of variation of inter-tap intervals (rhythm/freezing/arrests).

        Criteria:
            - 0 (Normal): Regular rhythm (rhythm_cv < 0.20), rate >= 2.5 taps/s,
              no significant amplitude decay (slope > -0.015).
            - 1 (Slight): Any of: rate 2.0-2.5 taps/s, mild decay (-0.035 < slope <= -0.015),
              or slight rhythm irregularity (0.20 <= rhythm_cv < 0.30).
            - 2 (Mild): Any of: rate 1.5-2.0 taps/s, moderate decay (-0.06 < slope <= -0.035),
              or rhythm irregularity (0.30 <= rhythm_cv < 0.45).
            - 3 (Moderate): Rate 1.0-1.5 taps/s, severe decay (slope <= -0.06),
              or frequent interruptions/arrests (rhythm_cv >= 0.45).
            - 4 (Severe): Barely able to perform, rate < 1.0 taps/s or unable to
              sustain consecutive taps (e.g. total_taps < 3).

        Args:
            tap_metrics: Dictionary produced by FingerTappingAnalyzer containing
                         'total_taps', 'tap_rate_hz', 'amplitude_decay_slope',
                         'rhythm_cv', and 'mean_amplitude'.

        Returns:
            Dictionary containing:
                - 'score' (int): MDS-UPDRS rating [0, 4].
                - 'label' (str): Descriptive severity label.
                - 'clinical_rationale' (str): Explanatory clinical summary.
                - 'domain_breakdown' (Dict[str, int]): Individual ratings for rate, decay, and rhythm.
        """
        rate = float(tap_metrics.get("tap_rate_hz", 0.0))
        slope = float(tap_metrics.get("amplitude_decay_slope", 0.0))
        cv = float(tap_metrics.get("rhythm_cv", 0.0))
        total_taps = int(tap_metrics.get("total_taps", 0))

        # Edge case: patient unable to sustain consecutive taps
        if total_taps < 3:
            return {
                "score": 4,
                "label": cls.SCORE_LABELS[4],
                "clinical_rationale": (
                    f"Severe motor impairment (Score 4): Unable to sustain consecutive finger taps "
                    f"(detected {total_taps} taps, rate: {rate:.1f} Hz)."
                ),
                "domain_breakdown": {
                    "rate_score": 4,
                    "decay_score": 4 if total_taps == 0 else 0,
                    "rhythm_score": 4 if total_taps == 0 else 0,
                },
            }

        # 1. Evaluate Tap Rate (Bradykinesia / Speed)
        if rate < 1.0:
            rate_score = 4
            rate_desc = f"severe slowing ({rate:.2f} taps/s < 1.0 taps/s)"
        elif 1.0 <= rate < 1.5:
            rate_score = 3
            rate_desc = f"moderate slowing ({rate:.2f} taps/s)"
        elif 1.5 <= rate < 2.0:
            rate_score = 2
            rate_desc = f"mild slowing ({rate:.2f} taps/s)"
        elif 2.0 <= rate < 2.5:
            rate_score = 1
            rate_desc = f"slight slowing ({rate:.2f} taps/s)"
        else:
            rate_score = 0
            rate_desc = f"normal tapping rate ({rate:.2f} taps/s >= 2.5 taps/s)"

        # 2. Evaluate Amplitude Decay (Decrement / Fatigue)
        if slope <= -0.06:
            decay_score = 3
            decay_desc = f"severe amplitude decrement (slope: {slope:.3f} <= -0.060)"
        elif -0.06 < slope <= -0.035:
            decay_score = 2
            decay_desc = f"moderate amplitude decrement (slope: {slope:.3f})"
        elif -0.035 < slope <= -0.015:
            decay_score = 1
            decay_desc = f"mild amplitude decrement (slope: {slope:.3f})"
        else:
            decay_score = 0
            decay_desc = f"sustained amplitude (slope: {slope:.3f} > -0.015)"

        # 3. Evaluate Rhythm Irregularity (Hesitations / Freezing / Arrests)
        if cv >= 0.45:
            rhythm_score = 3
            rhythm_desc = f"frequent interruptions/arrests (rhythm CV: {cv:.2f} >= 0.45)"
        elif 0.30 <= cv < 0.45:
            rhythm_score = 2
            rhythm_desc = f"moderate rhythm irregularity (rhythm CV: {cv:.2f})"
        elif 0.20 <= cv < 0.30:
            rhythm_score = 1
            rhythm_desc = f"slight rhythm irregularity (rhythm CV: {cv:.2f})"
        else:
            rhythm_score = 0
            rhythm_desc = f"regular rhythm (rhythm CV: {cv:.2f} < 0.20)"

        # Overall MDS-UPDRS rating is determined by the most severe limiting factor
        overall_score = max(rate_score, decay_score, rhythm_score)

        # Build clinical rationale statement
        if overall_score == 0:
            rationale = (
                "Normal motor performance (Score 0): Regular tapping rhythm (CV < 0.20), "
                f"brisk tapping rate ({rate:.2f} taps/s), and sustained amplitude without decrement."
            )
        else:
            reasons: List[str] = []
            if rate_score == overall_score:
                reasons.append(rate_desc)
            if decay_score == overall_score:
                reasons.append(decay_desc)
            if rhythm_score == overall_score:
                reasons.append(rhythm_desc)

            rationale = (
                f"{cls.SCORE_LABELS[overall_score]} impairment (Score {overall_score}): "
                f"Characterized by {', and '.join(reasons)}."
            )

        return {
            "score": int(overall_score),
            "label": cls.SCORE_LABELS[overall_score],
            "clinical_rationale": rationale,
            "domain_breakdown": {
                "rate_score": int(rate_score),
                "decay_score": int(decay_score),
                "rhythm_score": int(rhythm_score),
            },
        }

    @classmethod
    def score_tremor(cls, tremor_metrics: Dict[str, Any]) -> Dict[str, Any]:
        """
        Score MDS-UPDRS Part III (Tasks 3.15 - 3.18 - Tremor Assessment).

        Maps tremor power and dominant frequency to 0-4 severity scale.

        Scale:
            - 0 (Normal): No significant tremor oscillation detected.
            - 1 (Slight): Low-amplitude tremor (peak power in [0.001, 0.01)).
            - 2 (Mild): Moderate-low tremor (peak power in [0.01, 0.05)).
            - 3 (Moderate): Prominent tremor oscillation (peak power in [0.05, 0.15)).
            - 4 (Severe): High-amplitude disruptive tremor (peak power >= 0.15).

        Args:
            tremor_metrics: Dictionary containing 'peak_freq', 'peak_power',
                            'tremor_type', and 'is_significant'.

        Returns:
            Dictionary containing 'score', 'label', and 'clinical_rationale'.
        """
        peak_freq = float(tremor_metrics.get("peak_freq", 0.0))
        peak_power = float(tremor_metrics.get("peak_power", 0.0))
        tremor_type = str(tremor_metrics.get("tremor_type", "Voluntary baseline movement"))
        is_significant = bool(tremor_metrics.get("is_significant", False))

        if not is_significant or peak_freq < 3.0 or peak_power < 0.001:
            return {
                "score": 0,
                "label": cls.SCORE_LABELS[0],
                "clinical_rationale": (
                    "Normal (Score 0): No pathological tremor oscillation detected in the 3.0-12.0 Hz band."
                ),
            }

        if peak_power < 0.01:
            score = 1
            label = cls.SCORE_LABELS[1]
            rationale = (
                f"Slight tremor (Score 1): Low-amplitude oscillation at {peak_freq:.1f} Hz "
                f"({tremor_type}, PSD power: {peak_power:.4f})."
            )
        elif peak_power < 0.05:
            score = 2
            label = cls.SCORE_LABELS[2]
            rationale = (
                f"Mild tremor (Score 2): Moderate-low amplitude oscillation at {peak_freq:.1f} Hz "
                f"({tremor_type}, PSD power: {peak_power:.4f})."
            )
        elif peak_power < 0.15:
            score = 3
            label = cls.SCORE_LABELS[3]
            rationale = (
                f"Moderate tremor (Score 3): Prominent oscillation at {peak_freq:.1f} Hz "
                f"({tremor_type}, PSD power: {peak_power:.4f})."
            )
        else:
            score = 4
            label = cls.SCORE_LABELS[4]
            rationale = (
                f"Severe tremor (Score 4): High-amplitude disruptive oscillation at {peak_freq:.1f} Hz "
                f"({tremor_type}, PSD power: {peak_power:.4f})."
            )

        return {
            "score": int(score),
            "label": label,
            "clinical_rationale": rationale,
        }

    @staticmethod
    def score_rest_tremor(amplitude_cm: float) -> int:
        """
        Score MDS-UPDRS Item 3.17 (Rest Tremor Amplitude in centimeters).
        Provided for legacy compatibility.
        """
        amp = float(amplitude_cm)
        if amp < 0.5:
            return 0
        elif amp < 1.0:
            return 1
        elif amp < 3.0:
            return 2
        elif amp < 10.0:
            return 3
        else:
            return 4
