import pytest

from pearch.schema import V1ProfileResponse, V2SearchResponse


@pytest.mark.parametrize("response_model", [V1ProfileResponse, V2SearchResponse])
def test_profile_location_fields_survive_response_parsing(response_model):
    profile = {
        "location": "Bristol, England, United Kingdom",
        "location_coordinates": {"lng": -2.6, "lat": 51.5},
        "location_country_code": "GB",
        "location_regions": ["Europe", "EMEA"],
    }
    if response_model is V1ProfileResponse:
        parsed_profile = response_model.model_validate({"profile": profile}).profile
    else:
        response = response_model.model_validate({"search_results": [{"docid": "bristol", "profile": profile}]})
        parsed_profile = response.search_results[0].profile

    assert parsed_profile.model_dump(exclude_none=True, include=set(profile)) == profile
