from pearch import (
    SearchRequirement,
    V2SearchCountBucket,
    V2SearchCountFieldStats,
    V2SearchCountRequest,
    V2SearchCountResponse,
    V2SearchCountStats,
    V2SearchCountTopValue,
)


def test_search_count_request_serializes_query_requirements_and_stats():
    request = V2SearchCountRequest(
        query="Python engineers",
        search_requirements=[
            SearchRequirement(
                search_requirement="Located in Germany",
                must_have=True,
            )
        ],
        custom_filters={"min_total_experience_years": 5},
        stats=True,
    )

    assert request.model_dump(exclude_none=True) == {
        "query": "Python engineers",
        "search_requirements": [
            {
                "search_requirement": "Located in Germany",
                "must_have": True,
            }
        ],
        "custom_filters": {"min_total_experience_years": 5.0},
        "stats": True,
    }


def test_search_count_response_parses_typed_stats():
    response = V2SearchCountResponse.model_validate(
        {
            "count": 5000,
            "approximate": False,
            "credits_used": 10,
            "credits_remaining": 90,
            "stats": {
                "matched_profiles": 5000,
                "sample_limit": 1000,
                "top_values_limit": 10,
                "sample_size": 1000,
                "truncated": True,
                "fields": {
                    "canonical_skills": {
                        "profiles_total": 1000,
                        "profiles_non_empty": 900,
                        "profiles_empty": 100,
                        "coverage_pct": 90,
                        "values_total": 4200,
                        "unique_values": 350,
                        "top_values": [
                            {
                                "value": "Python",
                                "profile_count": 400,
                                "percentage": 40,
                            }
                        ],
                    },
                    "estimated_age": {
                        "profiles_total": 1000,
                        "profiles_non_empty": 800,
                        "profiles_empty": 200,
                        "coverage_pct": 80,
                        "min": 20,
                        "max": 67,
                        "mean": 35.6,
                        "median": 34,
                        "buckets": [
                            {
                                "label": "30-34",
                                "profile_count": 320,
                                "percentage": 32,
                            }
                        ],
                    },
                    "decision_maker": {
                        "profiles_total": 1000,
                        "profiles_non_empty": 950,
                        "profiles_empty": 50,
                        "coverage_pct": 95,
                        "counts": {"true": 300, "false": 650, "unknown": 50},
                    },
                },
            },
        }
    )

    assert response.credits_used == 10
    assert isinstance(response.stats, V2SearchCountStats)
    assert response.stats.truncated is True

    skills = response.stats.fields["canonical_skills"]
    assert isinstance(skills, V2SearchCountFieldStats)
    assert skills.unique_values == 350
    assert isinstance(skills.top_values[0], V2SearchCountTopValue)
    assert skills.top_values[0].value == "Python"

    age = response.stats.fields["estimated_age"]
    assert isinstance(age.buckets[0], V2SearchCountBucket)
    assert age.mean == 35.6

    decision_maker = response.stats.fields["decision_maker"]
    assert decision_maker.counts == {"true": 300, "false": 650, "unknown": 50}
