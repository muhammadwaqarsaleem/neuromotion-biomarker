"""
HL7 FHIR R4 Compliant Data Export Module
=========================================
Serializes kinematic biomarkers and MDS-UPDRS clinical scores into
standardized HL7 FHIR R4 resources (Patient, Observation, DiagnosticReport,
and Bundle) for Electronic Health Record (EHR) integration.

Standards Compliance:
    - HL7 FHIR R4 (v4.0.1) Specification
    - LOINC (Logical Observation Identifiers Names and Codes)
    - SNOMED CT (Systematized Nomenclature of Medicine -- Clinical Terms)
    - UCUM (Unified Code for Units of Measure)
"""

import json
import uuid
from datetime import datetime, timezone
from typing import Any, Dict, Optional, Tuple, Union


class FHIRExporter:
    """
    Serializes kinematic measurements and MDS-UPDRS clinical scores into HL7 FHIR R4 resources.

    Attributes:
        patient_id (str): Unique identifier for the subject/patient.
    """

    def __init__(self, patient_id: str = "PATIENT-001") -> None:
        """
        Initialize the FHIRExporter instance.

        Args:
            patient_id: Target patient identifier.
        """
        self.patient_id = str(patient_id)

    def build_patient_resource(
        self,
        patient_meta: Dict[str, Any],
        patient_uuid: Optional[str] = None,
    ) -> Dict[str, Any]:
        """
        Create an HL7 FHIR R4 Patient resource.

        Args:
            patient_meta: Metadata dictionary (id, age, gender, name).
            patient_uuid: Optional UUID identifier for resource.

        Returns:
            Dictionary matching FHIR R4 Patient resource schema.
        """
        p_id = str(patient_meta.get("id") or patient_meta.get("patient_id") or self.patient_id)
        res_id = patient_uuid or str(uuid.uuid4())
        gender = str(patient_meta.get("gender", "unknown")).lower()
        if gender not in {"male", "female", "other", "unknown"}:
            gender = "unknown"

        resource: Dict[str, Any] = {
            "resourceType": "Patient",
            "id": res_id,
            "meta": {
                "profile": ["http://hl7.org/fhir/StructureDefinition/Patient"],
            },
            "identifier": [
                {
                    "use": "usual",
                    "system": "https://neuromotion.ai/patients",
                    "value": p_id,
                }
            ],
            "active": True,
            "name": [
                {
                    "use": "anonymous",
                    "text": str(patient_meta.get("name", f"Patient {p_id}")),
                }
            ],
            "gender": gender,
        }

        # If age is specified, add birthDate estimation or custom extension
        age = patient_meta.get("age")
        if age is not None:
            try:
                age_int = int(age)
                approx_year = datetime.now(timezone.utc).year - age_int
                resource["birthDate"] = f"{approx_year}-01-01"
                resource["extension"] = [
                    {
                        "url": "https://neuromotion.ai/fhir/StructureDefinition/patient-age",
                        "valueInteger": age_int,
                    }
                ]
            except (ValueError, TypeError):
                pass

        return resource

    def build_observation_resource(
        self,
        metric_name: str,
        value: Union[float, int],
        unit: str,
        code_loinc: str = "89555-7",
        code_snomed: Optional[str] = None,
        patient_uuid: Optional[str] = None,
        timestamp_iso: Optional[str] = None,
        components: Optional[list] = None,
        interpretation_label: Optional[str] = None,
    ) -> Dict[str, Any]:
        """
        Create an HL7 FHIR R4 Observation resource dictionary for a biomarker metric.

        Args:
            metric_name: Human readable name of the observation.
            value: Numerical metric value or score.
            unit: Physical unit (e.g., 'Hz', '1/s', '{score}', '%').
            code_loinc: Primary LOINC concept code.
            code_snomed: Optional SNOMED CT concept code.
            patient_uuid: UUID of the patient subject.
            timestamp_iso: ISO 8601 recording timestamp.
            components: Optional list of sub-components for composite observations.
            interpretation_label: Optional severity label ("Normal", "Mild", etc.).

        Returns:
            Dictionary formatted according to HL7 FHIR R4 Observation specification.
        """
        obs_id = str(uuid.uuid4())
        effective_time = timestamp_iso or datetime.now(timezone.utc).isoformat()

        coding = [
            {
                "system": "http://loinc.org",
                "code": code_loinc,
                "display": metric_name,
            }
        ]
        if code_snomed:
            coding.append(
                {
                    "system": "http://snomed.info/sct",
                    "code": code_snomed,
                    "display": metric_name,
                }
            )

        resource: Dict[str, Any] = {
            "resourceType": "Observation",
            "id": obs_id,
            "meta": {
                "profile": ["http://hl7.org/fhir/StructureDefinition/Observation"],
            },
            "status": "final",
            "category": [
                {
                    "coding": [
                        {
                            "system": "http://terminology.hl7.org/CodeSystem/observation-category",
                            "code": "exam",
                            "display": "Exam",
                        }
                    ],
                    "text": "Motor Examination",
                }
            ],
            "code": {
                "coding": coding,
                "text": metric_name,
            },
            "effectiveDateTime": effective_time,
        }

        if patient_uuid:
            resource["subject"] = {
                "reference": f"urn:uuid:{patient_uuid}",
            }

        if isinstance(value, int) or (isinstance(value, float) and value.is_integer() and unit == "{score}"):
            resource["valueInteger"] = int(value)
        else:
            resource["valueQuantity"] = {
                "value": float(value),
                "unit": unit,
                "system": "http://unitsofmeasure.org",
                "code": unit,
            }

        if interpretation_label:
            # FHIR standard interpretation code mapping
            interp_map = {
                "Normal": "N",
                "Slight": "A",
                "Mild": "A",
                "Moderate": "A",
                "Severe": "AA",
            }
            code_val = interp_map.get(interpretation_label, "A")
            resource["interpretation"] = [
                {
                    "coding": [
                        {
                            "system": "http://terminology.hl7.org/CodeSystem/v3-ObservationInterpretation",
                            "code": code_val,
                            "display": interpretation_label,
                        }
                    ],
                    "text": interpretation_label,
                }
            ]

        if components:
            resource["component"] = components

        return resource

    def export_bundle(
        self,
        patient_meta: Dict[str, Any],
        session_metrics: Dict[str, Any],
        clinical_scores: Dict[str, Any],
        bundle_type: str = "document",
    ) -> Dict[str, Any]:
        """
        Generate complete HL7 FHIR R4 Bundle.
        """
        return generate_fhir_diagnostic_report(
            patient_meta=patient_meta,
            session_metrics=session_metrics,
            clinical_scores=clinical_scores,
            bundle_type=bundle_type,
        )


def generate_fhir_diagnostic_report(
    patient_meta: Dict[str, Any],
    session_metrics: Dict[str, Any],
    clinical_scores: Dict[str, Any],
    bundle_type: str = "document",
) -> Dict[str, Any]:
    """
    Produce a valid, production-grade HL7 FHIR R4 Bundle containing Patient,
    MDS-UPDRS Task 3.4 (Finger Tapping) Observation, Task 3.15 (Tremor) Observation,
    and linking DiagnosticReport.

    Args:
        patient_meta: Dictionary containing patient details:
            - 'id' / 'patient_id': Anonymous identifier.
            - 'age': Patient age in years.
            - 'gender': 'male', 'female', 'other', or 'unknown'.
            - 'timestamp': Optional session ISO timestamp.
            - 'clinician_notes': Free-text clinical evaluation notes.
            - 'session_type': E.g. 'Clinic Visit', 'Telehealth Remote Session'.
        session_metrics: Dictionary containing raw kinematic measurements:
            - 'tapping' (or flattened): total_taps, tap_rate_hz, amplitude_decay_slope, rhythm_cv.
            - 'tremor' (or flattened): peak_freq, peak_power, tremor_type, is_significant.
        clinical_scores: Dictionary containing evaluated MDS-UPDRS scores:
            - 'tapping': score (0-4), label, clinical_rationale.
            - 'tremor': score (0-4), label, clinical_rationale.
        bundle_type: FHIR Bundle type ("document" or "transaction", default: "document").

    Returns:
        Formatted Python dictionary compliant with HL7 FHIR R4 Bundle specification.
    """
    bundle_uuid = str(uuid.uuid4())
    patient_uuid = str(uuid.uuid4())
    obs_tapping_uuid = str(uuid.uuid4())
    obs_tremor_uuid = str(uuid.uuid4())
    report_uuid = str(uuid.uuid4())

    session_time = (
        patient_meta.get("timestamp")
        or datetime.now(timezone.utc).isoformat()
    )

    # -------------------------------------------------------------------------
    # 1. Patient Resource
    # -------------------------------------------------------------------------
    exporter = FHIRExporter(patient_id=str(patient_meta.get("id") or patient_meta.get("patient_id") or "PT-1042"))
    patient_res = exporter.build_patient_resource(patient_meta, patient_uuid=patient_uuid)

    # -------------------------------------------------------------------------
    # 2. Extract Data Safely (Supporting both nested and flattened dictionaries)
    # -------------------------------------------------------------------------
    tap_metrics = session_metrics.get("tapping", session_metrics)
    tremor_metrics = session_metrics.get("tremor", session_metrics)

    tap_score_dict = clinical_scores.get("tapping", clinical_scores)
    tremor_score_dict = clinical_scores.get("tremor", clinical_scores)

    total_taps = int(tap_metrics.get("total_taps", 0))
    tap_rate = float(tap_metrics.get("tap_rate_hz", 0.0))
    decay_slope = float(tap_metrics.get("amplitude_decay_slope", 0.0))
    rhythm_cv = float(tap_metrics.get("rhythm_cv", 0.0))
    mean_amplitude = float(tap_metrics.get("mean_amplitude", 0.0))

    tap_score = int(tap_score_dict.get("score", 0))
    tap_label = str(tap_score_dict.get("label", "Normal"))
    tap_rationale = str(tap_score_dict.get("clinical_rationale", "Normal tapping motor assessment."))

    peak_freq = float(tremor_metrics.get("peak_freq", 0.0))
    peak_power = float(tremor_metrics.get("peak_power", 0.0))
    tremor_type = str(tremor_metrics.get("tremor_type", "Voluntary baseline movement"))
    is_significant = bool(tremor_metrics.get("is_significant", False))

    tremor_score = int(tremor_score_dict.get("score", 0))
    tremor_label = str(tremor_score_dict.get("label", "Normal"))
    tremor_rationale = str(tremor_score_dict.get("clinical_rationale", "No pathological tremor detected."))

    # -------------------------------------------------------------------------
    # 3. Observation: MDS-UPDRS Finger Tapping (Task 3.4)
    # -------------------------------------------------------------------------
    tapping_components = [
        {
            "code": {
                "coding": [
                    {
                        "system": "https://neuromotion.ai/biomarkers",
                        "code": "total-taps",
                        "display": "Total Confirmed Taps",
                    }
                ],
                "text": "Total Taps Count",
            },
            "valueQuantity": {
                "value": total_taps,
                "unit": "taps",
                "system": "http://unitsofmeasure.org",
                "code": "{taps}",
            },
        },
        {
            "code": {
                "coding": [
                    {
                        "system": "http://loinc.org",
                        "code": "89556-5",
                        "display": "Tapping rate",
                    },
                    {
                        "system": "https://neuromotion.ai/biomarkers",
                        "code": "mean-cadence",
                        "display": "Mean Tapping Cadence",
                    },
                ],
                "text": "Mean Cadence",
            },
            "valueQuantity": {
                "value": round(tap_rate, 2),
                "unit": "Hz",
                "system": "http://unitsofmeasure.org",
                "code": "Hz",
            },
        },
        {
            "code": {
                "coding": [
                    {
                        "system": "https://neuromotion.ai/biomarkers",
                        "code": "amplitude-decay-slope",
                        "display": "Amplitude Decrement / Fatigue Slope",
                    }
                ],
                "text": "Amplitude Decay Slope",
            },
            "valueQuantity": {
                "value": round(decay_slope, 4),
                "unit": "1/s",
                "system": "http://unitsofmeasure.org",
                "code": "1/s",
            },
        },
        {
            "code": {
                "coding": [
                    {
                        "system": "https://neuromotion.ai/biomarkers",
                        "code": "rhythm-cv",
                        "display": "Tap Interval Rhythm Coefficient of Variation",
                    }
                ],
                "text": "Rhythm Regularity CV",
            },
            "valueQuantity": {
                "value": round(rhythm_cv * 100.0, 2),
                "unit": "%",
                "system": "http://unitsofmeasure.org",
                "code": "%",
            },
        },
        {
            "code": {
                "coding": [
                    {
                        "system": "https://neuromotion.ai/biomarkers",
                        "code": "mean-aperture",
                        "display": "Mean Normalized Aperture",
                    }
                ],
                "text": "Mean Normalized Aperture",
            },
            "valueQuantity": {
                "value": round(mean_amplitude, 4),
                "unit": "{scale_ratio}",
                "system": "https://neuromotion.ai/units",
                "code": "{ratio}",
            },
        },
    ]

    obs_tapping_res: Dict[str, Any] = {
        "resourceType": "Observation",
        "id": obs_tapping_uuid,
        "meta": {
            "profile": ["http://hl7.org/fhir/StructureDefinition/Observation"],
        },
        "status": "final",
        "category": [
            {
                "coding": [
                    {
                        "system": "http://terminology.hl7.org/CodeSystem/observation-category",
                        "code": "exam",
                        "display": "Exam",
                    }
                ],
                "text": "Motor Examination",
            }
        ],
        "code": {
            "coding": [
                {
                    "system": "http://loinc.org",
                    "code": "89555-7",
                    "display": "MDS-UPDRS Part III - Finger tapping [Score]",
                },
                {
                    "system": "http://snomed.info/sct",
                    "code": "285640004",
                    "display": "Finger tapping test (procedure)",
                },
            ],
            "text": "MDS-UPDRS Task 3.4 Finger Tapping Motor Assessment",
        },
        "subject": {
            "reference": f"urn:uuid:{patient_uuid}",
            "display": str(patient_meta.get("name", f"Patient {exporter.patient_id}")),
        },
        "effectiveDateTime": session_time,
        "valueInteger": tap_score,
        "interpretation": [
            {
                "coding": [
                    {
                        "system": "http://terminology.hl7.org/CodeSystem/v3-ObservationInterpretation",
                        "code": "N" if tap_score == 0 else "A",
                        "display": tap_label,
                    }
                ],
                "text": f"MDS-UPDRS Rating: {tap_score} ({tap_label})",
            }
        ],
        "note": [{"text": tap_rationale}],
        "component": tapping_components,
    }

    # -------------------------------------------------------------------------
    # 4. Observation: Postural / Rest Tremor (Task 3.15)
    # -------------------------------------------------------------------------
    tremor_components = [
        {
            "code": {
                "coding": [
                    {
                        "system": "http://loinc.org",
                        "code": "89554-0",
                        "display": "Tremor frequency",
                    },
                    {
                        "system": "https://neuromotion.ai/biomarkers",
                        "code": "peak-tremor-frequency",
                        "display": "Dominant Peak Tremor Frequency",
                    },
                ],
                "text": "Dominant Tremor Peak Frequency",
            },
            "valueQuantity": {
                "value": round(peak_freq, 2),
                "unit": "Hz",
                "system": "http://unitsofmeasure.org",
                "code": "Hz",
            },
        },
        {
            "code": {
                "coding": [
                    {
                        "system": "https://neuromotion.ai/biomarkers",
                        "code": "spectral-peak-power",
                        "display": "Spectral Peak Power (PSD Magnitude)",
                    }
                ],
                "text": "Spectral Peak Power",
            },
            "valueQuantity": {
                "value": round(peak_power, 6),
                "unit": "{psd_magnitude}",
                "system": "https://neuromotion.ai/units",
                "code": "PSD",
            },
        },
        {
            "code": {
                "coding": [
                    {
                        "system": "https://neuromotion.ai/biomarkers",
                        "code": "tremor-classification",
                        "display": "Tremor Frequency Band Classification",
                    }
                ],
                "text": "Tremor Classification",
            },
            "valueString": tremor_type,
        },
    ]

    obs_tremor_res: Dict[str, Any] = {
        "resourceType": "Observation",
        "id": obs_tremor_uuid,
        "meta": {
            "profile": ["http://hl7.org/fhir/StructureDefinition/Observation"],
        },
        "status": "final",
        "category": [
            {
                "coding": [
                    {
                        "system": "http://terminology.hl7.org/CodeSystem/observation-category",
                        "code": "exam",
                        "display": "Exam",
                    }
                ],
                "text": "Tremor Assessment",
            }
        ],
        "code": {
            "coding": [
                {
                    "system": "http://loinc.org",
                    "code": "89553-2",
                    "display": "MDS-UPDRS Part III - Postural tremor of hands [Score]",
                },
                {
                    "system": "http://snomed.info/sct",
                    "code": "247545004",
                    "display": "Tremor (finding)",
                },
            ],
            "text": "MDS-UPDRS Task 3.15 Tremor Kinematic Assessment",
        },
        "subject": {
            "reference": f"urn:uuid:{patient_uuid}",
            "display": str(patient_meta.get("name", f"Patient {exporter.patient_id}")),
        },
        "effectiveDateTime": session_time,
        "valueInteger": tremor_score,
        "interpretation": [
            {
                "coding": [
                    {
                        "system": "http://terminology.hl7.org/CodeSystem/v3-ObservationInterpretation",
                        "code": "N" if tremor_score == 0 else "A",
                        "display": tremor_label,
                    }
                ],
                "text": f"MDS-UPDRS Tremor Rating: {tremor_score} ({tremor_label})",
            }
        ],
        "note": [{"text": tremor_rationale}],
        "component": tremor_components,
    }

    # -------------------------------------------------------------------------
    # 5. DiagnosticReport Linking Observations
    # -------------------------------------------------------------------------
    clinician_notes = patient_meta.get("clinician_notes", "Routine digital biomarker evaluation completed.")
    session_type = patient_meta.get("session_type", "Telehealth Motor Examination")

    report_res: Dict[str, Any] = {
        "resourceType": "DiagnosticReport",
        "id": report_uuid,
        "meta": {
            "profile": ["http://hl7.org/fhir/StructureDefinition/DiagnosticReport"],
        },
        "status": "final",
        "category": [
            {
                "coding": [
                    {
                        "system": "http://terminology.hl7.org/CodeSystem/v2-0074",
                        "code": "NEUR",
                        "display": "Neurology",
                    }
                ],
                "text": "Neurological Motor Assessment",
            }
        ],
        "code": {
            "coding": [
                {
                    "system": "http://loinc.org",
                    "code": "LP29684-5",
                    "display": "Neurology Diagnostic report",
                },
                {
                    "system": "http://loinc.org",
                    "code": "11502-2",
                    "display": "Laboratory report",
                },
            ],
            "text": f"NeuroMotion Motor Biomarker Report - {session_type}",
        },
        "subject": {
            "reference": f"urn:uuid:{patient_uuid}",
            "display": str(patient_meta.get("name", f"Patient {exporter.patient_id}")),
        },
        "effectiveDateTime": session_time,
        "issued": datetime.now(timezone.utc).isoformat(),
        "performer": [
            {
                "display": str(patient_meta.get("clinician_name", "NeuroMotion AI Edge Engine")),
            }
        ],
        "result": [
            {
                "reference": f"urn:uuid:{obs_tapping_uuid}",
                "display": f"Task 3.4 Finger Tapping (Score: {tap_score})",
            },
            {
                "reference": f"urn:uuid:{obs_tremor_uuid}",
                "display": f"Task 3.15 Tremor Assessment (Score: {tremor_score})",
            },
        ],
        "conclusion": f"Finger Tapping: Score {tap_score} ({tap_label}). Tremor: Score {tremor_score} ({tremor_label}).",
        "note": [
            {
                "text": f"Clinician Notes: {clinician_notes}",
            },
            {
                "text": "Assistive AI disclaimer: Clinician-in-the-loop assistive biomarker evaluation. Not an autonomous diagnostic verdict.",
            },
        ],
    }

    # -------------------------------------------------------------------------
    # 6. FHIR R4 Bundle Assembly
    # -------------------------------------------------------------------------
    bundle: Dict[str, Any] = {
        "resourceType": "Bundle",
        "id": bundle_uuid,
        "meta": {
            "profile": [
                f"http://hl7.org/fhir/StructureDefinition/{bundle_type}"
            ],
            "lastUpdated": datetime.now(timezone.utc).isoformat(),
        },
        "identifier": {
            "system": "https://neuromotion.ai/fhir/bundles",
            "value": f"bundle-{bundle_uuid[:8]}",
        },
        "type": bundle_type,
        "timestamp": session_time,
        "entry": [
            {
                "fullUrl": f"urn:uuid:{report_uuid}",
                "resource": report_res,
            },
            {
                "fullUrl": f"urn:uuid:{patient_uuid}",
                "resource": patient_res,
            },
            {
                "fullUrl": f"urn:uuid:{obs_tapping_uuid}",
                "resource": obs_tapping_res,
            },
            {
                "fullUrl": f"urn:uuid:{obs_tremor_uuid}",
                "resource": obs_tremor_res,
            },
        ],
    }

    return bundle


def export_fhir_json(
    patient_meta: Dict[str, Any],
    session_metrics: Dict[str, Any],
    clinical_scores: Dict[str, Any],
    indent: int = 2,
    bundle_type: str = "document",
) -> Tuple[Dict[str, Any], str]:
    """
    Convenience method returning both the Python dictionary and indented JSON string
    ready for Streamlit download button or direct EHR dispatch.

    Args:
        patient_meta: Patient and clinician metadata.
        session_metrics: Digital biomarker measurements.
        clinical_scores: Evaluated MDS-UPDRS clinical scores.
        indent: JSON indentation spaces (default: 2).
        bundle_type: FHIR bundle type ("document" or "transaction").

    Returns:
        Tuple of (bundle_dict, json_serialized_string).
    """
    bundle_dict = generate_fhir_diagnostic_report(
        patient_meta=patient_meta,
        session_metrics=session_metrics,
        clinical_scores=clinical_scores,
        bundle_type=bundle_type,
    )
    json_str = json.dumps(bundle_dict, indent=indent, default=str)
    return bundle_dict, json_str
