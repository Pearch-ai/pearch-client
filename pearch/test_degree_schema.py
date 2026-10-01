from pathlib import Path

import pytest
import yaml
from pydantic import ValidationError

from pearch.schema import CustomFilters, V1ProfileResponse, V2SearchResponse


DEGREES = ["associate", "bachelor", "master", "MBA", "doctor", "postdoc"]


@pytest.mark.parametrize("degree", DEGREES)
def test_custom_filters_accept_supported_degrees(degree):
    assert CustomFilters(degrees=[degree]).model_dump(exclude_none=True) == {"degrees": [degree]}


def test_custom_filters_reject_unknown_degree():
    with pytest.raises(ValidationError):
        CustomFilters(degrees=["unknown-degree"])


@pytest.mark.parametrize("response_model", [V1ProfileResponse, V2SearchResponse])
def test_associate_degree_survives_profile_response_parsing(response_model):
    profile = {"educations": [{"major": "Associate of Science", "degree": ["associate"]}]}
    if response_model is V1ProfileResponse:
        parsed = response_model.model_validate({"profile": profile}).profile
    else:
        response = response_model.model_validate({"search_results": [{"docid": "associate-example", "profile": profile}]})
        parsed = response.search_results[0].profile
    assert parsed.educations[0].degree == ["associate"]
    assert parsed.educations[0].major == "Associate of Science"


def test_openapi_request_and_response_degree_enums_match_client():
    source = Path(__file__).resolve().parents[1] / "pearch-openapi.yaml"
    schemas = yaml.safe_load(source.read_text())["components"]["schemas"]
    filters = schemas["CustomFilters"]["properties"]
    for field in ("degrees", "not_degrees"):
        assert filters[field]["items"]["enum"] == DEGREES
    assert schemas["Education"]["properties"]["degree"]["items"]["enum"] == DEGREES
