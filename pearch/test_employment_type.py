import pytest

from pearch.schema import V1ProfileResponse, V2SearchResponse


@pytest.mark.parametrize("response_model", [V1ProfileResponse, V2SearchResponse])
@pytest.mark.parametrize(
    "employment_type", ["Contract", "Freelance", "Self-employed", "Full-time", None]
)
def test_role_employment_type_survives_response_parsing(response_model, employment_type):
    role = {"title": "Designer"}
    if employment_type is not None:
        role["employment_type"] = employment_type
    profile = {
        "experiences": [
            {"company_info": {"name": "Example"}, "company_roles": [role]}
        ]
    }
    if response_model is V1ProfileResponse:
        response = response_model.model_validate({"profile": profile})
        parsed_profile = response.profile
    else:
        response = response_model.model_validate(
            {"search_results": [{"docid": "employment-type", "profile": profile}]}
        )
        parsed_profile = response.search_results[0].profile

    parsed_role = parsed_profile.experiences[0].company_roles[0]
    assert parsed_role.employment_type == employment_type
    assert parsed_role.model_dump(exclude_none=True).get("employment_type") == employment_type
