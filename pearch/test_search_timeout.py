from pathlib import Path

import pytest
import yaml
from pydantic import ValidationError

from pearch.schema import V2SearchRequest


@pytest.mark.parametrize("parameter", ["time_budget", "timeout"])
def test_search_request_serializes_budget_synonyms(parameter):
    request = V2SearchRequest(query="engineers", type="fast", **{parameter: 45})
    assert request.model_dump(exclude_none=True)[parameter] == 45


@pytest.mark.parametrize("parameter", ["time_budget", "timeout"])
@pytest.mark.parametrize("value", [0, -1])
def test_search_request_rejects_non_positive_budgets(parameter, value):
    with pytest.raises(ValidationError):
        V2SearchRequest(query="engineers", **{parameter: value})


@pytest.mark.parametrize("path", ["/v2/search", "/v2/search/submit"])
def test_openapi_documents_both_budget_names(path):
    source = Path(__file__).resolve().parents[1] / "pearch-openapi.yaml"
    operation = yaml.safe_load(source.read_text())["paths"][path]["post"]
    properties = operation["requestBody"]["content"]["application/json"]["schema"]["properties"]
    for parameter in ("time_budget", "timeout"):
        assert properties[parameter]["type"] == "integer"
        assert properties[parameter]["minimum"] == 1
        assert "time_budget takes precedence" in properties[parameter]["description"]
