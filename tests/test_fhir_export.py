"""
Unit Tests for HL7 FHIR R4 Compliant Export
============================================
Verifies Patient, Observation (Task 3.4 & Task 3.15), DiagnosticReport,
and Bundle construction in clinical/fhir_export.py.
"""

import json
import unittest

from clinical.fhir_export import (
    FHIRExporter,
    generate_fhir_diagnostic_report,
    export_fhir_json,
)


class TestFHIRExport(unittest.TestCase):
    """Test suite for HL7 FHIR R4 bundle generation and serialization."""

    def setUp(self):
        self.patient_meta = {
            "id": "PT-90210",
            "name": "Jane Doe",
            "age": 67,
            "gender": "female",
            "clinician_name": "Dr. Sarah Adams, MD",
            "clinician_notes": "Mild bradykinesia noted during left finger tapping; no resting tremor.",
            "session_type": "Telehealth Remote Exam",
        }
        self.session_metrics = {
            "tapping": {
                "total_taps": 22,
                "tap_rate_hz": 2.2,
                "amplitude_decay_slope": -0.025,
                "rhythm_cv": 0.24,
                "mean_amplitude": 0.38,
            },
            "tremor": {
                "peak_freq": 5.2,
                "peak_power": 0.0035,
                "tremor_type": "Parkinsonian / Resting tremor",
                "is_significant": True,
            },
        }
        self.clinical_scores = {
            "tapping": {
                "score": 1,
                "label": "Slight",
                "clinical_rationale": "Slight impairment: Characterized by slight slowing (2.20 taps/s).",
            },
            "tremor": {
                "score": 1,
                "label": "Slight",
                "clinical_rationale": "Slight tremor: Low-amplitude oscillation at 5.2 Hz.",
            },
        }

    def test_generate_fhir_diagnostic_report_structure(self):
        bundle = generate_fhir_diagnostic_report(
            self.patient_meta, self.session_metrics, self.clinical_scores
        )

        self.assertEqual(bundle["resourceType"], "Bundle")
        self.assertEqual(bundle["type"], "document")
        self.assertIn("entry", bundle)
        self.assertEqual(len(bundle["entry"]), 4)

        # Map entries by resourceType
        resources = {entry["resource"]["resourceType"]: entry["resource"] for entry in bundle["entry"]}
        self.assertIn("Patient", resources)
        self.assertIn("DiagnosticReport", resources)

        # Observations
        obs_list = [entry["resource"] for entry in bundle["entry"] if entry["resource"]["resourceType"] == "Observation"]
        self.assertEqual(len(obs_list), 2)

    def test_patient_resource_attributes(self):
        bundle = generate_fhir_diagnostic_report(
            self.patient_meta, self.session_metrics, self.clinical_scores
        )
        patient_entry = next(e["resource"] for e in bundle["entry"] if e["resource"]["resourceType"] == "Patient")

        self.assertEqual(patient_entry["gender"], "female")
        self.assertIn("birthDate", patient_entry)
        self.assertEqual(patient_entry["identifier"][0]["value"], "PT-90210")
        self.assertEqual(patient_entry["name"][0]["text"], "Jane Doe")

    def test_tapping_observation_components(self):
        bundle = generate_fhir_diagnostic_report(
            self.patient_meta, self.session_metrics, self.clinical_scores
        )
        obs_list = [e["resource"] for e in bundle["entry"] if e["resource"]["resourceType"] == "Observation"]

        # Finger tapping observation
        tapping_obs = next(
            o for o in obs_list
            if any(c.get("code") == "89555-7" for c in o["code"]["coding"])
        )
        self.assertEqual(tapping_obs["valueInteger"], 1)
        self.assertEqual(tapping_obs["status"], "final")

        # Verify components: Total Taps, Cadence, Decay Slope, Rhythm CV
        components = {c["code"]["text"]: c for c in tapping_obs["component"]}
        self.assertIn("Total Taps Count", components)
        self.assertEqual(components["Total Taps Count"]["valueQuantity"]["value"], 22)

        self.assertIn("Mean Cadence", components)
        self.assertEqual(components["Mean Cadence"]["valueQuantity"]["unit"], "Hz")

        self.assertIn("Amplitude Decay Slope", components)
        self.assertEqual(components["Amplitude Decay Slope"]["valueQuantity"]["value"], -0.025)

        self.assertIn("Rhythm Regularity CV", components)
        self.assertEqual(components["Rhythm Regularity CV"]["valueQuantity"]["value"], 24.0)

    def test_tremor_observation(self):
        bundle = generate_fhir_diagnostic_report(
            self.patient_meta, self.session_metrics, self.clinical_scores
        )
        obs_list = [e["resource"] for e in bundle["entry"] if e["resource"]["resourceType"] == "Observation"]

        tremor_obs = next(
            o for o in obs_list
            if any(c.get("code") == "89553-2" for c in o["code"]["coding"])
        )
        self.assertEqual(tremor_obs["valueInteger"], 1)

        components = {c["code"]["text"]: c for c in tremor_obs["component"]}
        self.assertIn("Dominant Tremor Peak Frequency", components)
        self.assertEqual(components["Dominant Tremor Peak Frequency"]["valueQuantity"]["value"], 5.2)

        self.assertIn("Tremor Classification", components)
        self.assertEqual(components["Tremor Classification"]["valueString"], "Parkinsonian / Resting tremor")

    def test_diagnostic_report_and_disclaimer(self):
        bundle = generate_fhir_diagnostic_report(
            self.patient_meta, self.session_metrics, self.clinical_scores
        )
        report = next(e["resource"] for e in bundle["entry"] if e["resource"]["resourceType"] == "DiagnosticReport")

        self.assertEqual(report["status"], "final")
        self.assertEqual(len(report["result"]), 2)

        # Check disclaimer and clinician notes
        notes_text = " ".join([n["text"] for n in report.get("note", [])])
        self.assertIn("Assistive AI disclaimer", notes_text)
        self.assertIn("Not an autonomous diagnostic verdict", notes_text)
        self.assertIn("Mild bradykinesia noted", notes_text)

    def test_export_fhir_json_serialization(self):
        bundle_dict, json_str = export_fhir_json(
            self.patient_meta, self.session_metrics, self.clinical_scores, indent=2
        )
        self.assertIsInstance(bundle_dict, dict)
        self.assertIsInstance(json_str, str)

        # Verify valid JSON
        parsed = json.loads(json_str)
        self.assertEqual(parsed["resourceType"], "Bundle")
        self.assertEqual(len(parsed["entry"]), 4)

    def test_flattened_dictionary_input(self):
        # Test fallback when callers pass flattened dictionaries
        flat_metrics = {
            "total_taps": 15,
            "tap_rate_hz": 1.5,
            "amplitude_decay_slope": -0.04,
            "rhythm_cv": 0.35,
            "peak_freq": 6.0,
            "peak_power": 0.02,
        }
        flat_scores = {
            "score": 2,
            "label": "Mild",
            "clinical_rationale": "Mild impairment",
        }
        bundle = generate_fhir_diagnostic_report({"id": "PT-FLAT"}, flat_metrics, flat_scores)
        self.assertEqual(bundle["resourceType"], "Bundle")
        self.assertEqual(len(bundle["entry"]), 4)


if __name__ == "__main__":
    unittest.main()
