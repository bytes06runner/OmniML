from __future__ import annotations

import os

from anomallm.compliance import generate_compliance_reports


def test_generate_compliance_reports_eu_ai_act(sample_bundle):
    reports = generate_compliance_reports(sample_bundle, ["eu_ai_act"])
    assert len(reports) == 1
    report = reports[0]
    assert report.template_id == "eu_ai_act"
    assert report.markdown_path
    assert os.path.exists(report.markdown_path)
