import pytest
from pydantic import ValidationError

from pearch.schema import V1ProfileResponse, V2SearchRequest, V2SearchResponse


def test_search_request_sends_sources():
    request = V2SearchRequest(query="family nurse practitioner in California", sources=["healthcare"])

    assert request.model_dump(exclude_none=True, by_alias=True)["sources"] == ["healthcare"]


@pytest.mark.parametrize("sources", [[], ["registry"]])
def test_search_request_rejects_empty_or_unknown_sources(sources):
    with pytest.raises(ValidationError):
        V2SearchRequest(query="family nurse practitioner in California", sources=sources)


@pytest.mark.parametrize("response_model", [V1ProfileResponse, V2SearchResponse])
def test_registry_record_fields_survive_response_parsing(response_model):
    profile = {
        "docid": "npi:1023555992",
        "npi": "1023555992",
        "linkedin_slug": None,
        "title": "Family Nurse Practitioner",
        "sources": ["healthcare"],
    }
    if response_model is V1ProfileResponse:
        parsed_profile = response_model.model_validate({"profile": profile}).profile
    else:
        response = response_model.model_validate({"search_results": [{"docid": profile["docid"], "profile": profile}]})
        parsed_profile = response.search_results[0].profile

    assert parsed_profile.model_dump(include=set(profile)) == profile
