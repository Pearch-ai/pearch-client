from datetime import date
from pathlib import Path

import pytest
import yaml

from pearch.schema import V1ProfileResponse, V2SearchResponse

CERTIFICATION_FIELDS = ["title", "issuer", "issuer_url", "certificate_url", "begin_date", "end_date"]

PROFILE = {
    "certifications": [
        {
            "title": "Project Management Professional (PMP)",
            "issuer": "Project Management Institute",
            "certificate_url": "https://www.credly.com/badges/example",
            "begin_date": "2019-03-01",
            "end_date": "2024-03-01",
        },
        {"title": "Scrum Master"},
    ],
    "experiences": [{"company": "Example", "company_info": {"yc_batch": "W22", "yc_batch_year": 2022}}],
    "recommendations": [{"recommendation": "Jane mentored three engineers.", "referee_name": "John Smith"}],
    "activities": [{"title": "We cut Kubernetes costs by 40%", "action": "Posted by"}],
}


@pytest.mark.parametrize("response_model", [V1ProfileResponse, V2SearchResponse])
def test_certificate_details_and_yc_batch_survive_profile_response_parsing(response_model):
    if response_model is V1ProfileResponse:
        parsed = response_model.model_validate({"profile": PROFILE}).profile
    else:
        response = response_model.model_validate({"search_results": [{"docid": "pmp-holder", "profile": PROFILE}]})
        parsed = response.search_results[0].profile

    certificate, undated = parsed.certifications
    assert certificate.issuer == "Project Management Institute"
    assert certificate.certificate_url == "https://www.credly.com/badges/example"
    assert certificate.begin_date == date(2019, 3, 1)
    assert certificate.end_date == date(2024, 3, 1)
    assert undated.end_date is None
    assert parsed.experiences[0].company_info.yc_batch == "W22"
    assert parsed.experiences[0].company_info.yc_batch_year == 2022
    assert parsed.recommendations[0].referee_name == "John Smith"
    assert parsed.activities[0].action == "Posted by"


def test_openapi_documents_certificate_details_and_yc_batch():
    source = Path(__file__).resolve().parents[1] / "pearch-openapi.yaml"
    schemas = yaml.safe_load(source.read_text())["components"]["schemas"]

    assert list(schemas["Certification"]["properties"]) == CERTIFICATION_FIELDS
    company_info = next(
        schema["properties"]
        for schema in schemas.values()
        if "backed_by_y_combinator" in schema.get("properties", {})
    )
    assert company_info["yc_batch"]["type"] == "string"
    assert company_info["yc_batch_year"]["type"] == "integer"
    assert list(schemas["Recommendation"]["properties"]) == ["recommendation", "referee_name", "referee_url"]
    assert schemas["Activity"]["properties"]["action"]["enum"] == ["Posted by", "Shared by"]
    profile = next(schema["properties"] for schema in schemas.values() if "certifications" in schema.get("properties", {}))
    assert profile["recommendations"]["items"]["$ref"] == "#/components/schemas/Recommendation"
    assert profile["activities"]["items"]["$ref"] == "#/components/schemas/Activity"
