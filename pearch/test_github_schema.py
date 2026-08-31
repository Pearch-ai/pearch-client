import pytest
from pydantic import ValidationError

from pearch.schema import (
    Profile,
    V1ProfileRequest,
    V1ProfileResponse,
    V2GithubSearchRequest,
    V2GithubSearchResponse,
)


def test_profile_accepts_github_analysis_fields():
    profile = Profile(
        docid="jane-doe",
        github_handle="janedoe",
        github_skills=[{"skill": "rust", "weight": 9}],
        github_skills_curated=[
            {
                "skill": "Rust (Programming Language)",
                "canonical_skill": "Rust (Programming Language)",
                "weight": 9,
                "rank": 1,
                "score": 0.5,
            }
        ],
        github_oss_contributions=[{"project_name": "project"}],
    )

    assert profile.github_handle == "janedoe"
    assert profile.github_skills_curated[0]["rank"] == 1


def test_profile_request_accepts_github_id_as_identifier():
    request = V1ProfileRequest(github_id="abouhid2", with_profile=True)

    assert request.model_dump(exclude_none=True)["github_id"] == "abouhid2"


def test_profile_request_rejects_multiple_identifiers():
    with pytest.raises(ValidationError):
        V1ProfileRequest(docid="alexandrebouhid", github_id="abouhid2")


def test_profile_request_preserves_docid_email_compatibility():
    request = V1ProfileRequest(docid="alexandrebouhid", email="person@example.com")

    assert request.docid == "alexandrebouhid"
    assert request.email == "person@example.com"


def test_profile_response_preserves_credit_breakdown():
    response = V1ProfileResponse(
        profile={"github_handle": "abouhid2"},
        credits_breakdown={"version": 1, "total": 8, "items": []},
    )

    assert response.credits_breakdown["total"] == 8


def test_github_search_schema_preserves_raw_and_canonical_fields():
    request = V2GithubSearchRequest(query="rust engineer", limit=1)
    response = V2GithubSearchResponse(
        query=request.query,
        results=[
            {
                "profile": {
                    "github_handle": "janedoe",
                    "skills": [{"skill": "rust", "weight": 9}],
                    "github_skills": [{"skill": "rust", "weight": 9}],
                    "github_skills_curated": [
                        {
                            "skill": "Rust (Programming Language)",
                            "rank": 1,
                            "score": 0.5,
                        }
                    ],
                }
            }
        ],
        results_count=1,
        credits_used=8,
    )

    assert response.results[0].profile.skills[0]["skill"] == "rust"
    assert response.results[0].profile.github_skills_curated[0]["rank"] == 1


def test_github_search_schema_preserves_flat_results():
    response = V2GithubSearchResponse(
        query="rust engineer",
        results=[
            {
                "github_handle": "janedoe",
                "skills": [{"skill": "rust", "weight": 9}],
                "github_skills": [{"skill": "rust", "weight": 9}],
            }
        ],
        results_count=1,
    )

    assert response.results[0].profile.github_handle == "janedoe"
    assert response.results[0].profile.skills[0]["skill"] == "rust"
