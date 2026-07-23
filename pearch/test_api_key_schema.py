from pearch.schema import (
    ApiKeyMetadata,
    OrganizationMemberApiKeys,
    UserInfo,
    V1ApiCallHistoryRequest,
    V1ApiKeyCapabilitiesResponse,
    V1CreateApiKeyRequest,
    V1CreateOrganizationApiKeyResponse,
    V1OrganizationApiKeysResponse,
)


def test_owner_managed_api_key_schemas():
    metadata = ApiKeyMetadata(
        id="9ea5c01e-fac8-4aaf-9137-d2954afac230",
        name="Production",
        preview="pk_abc…1234",
        created_at="2026-07-20T12:00:00Z",
        last_used_at=None,
        revoked_at=None,
    )
    response = V1OrganizationApiKeysResponse(
        organization_id="1ad448f6-aa55-455f-a369-c26070b3fe80",
        members=[
            OrganizationMemberApiKeys(
                id="d3297401-bb9b-48c0-ae5e-d481159e6d4e",
                user_id=None,
                email="member@example.com",
                role="member",
                api_keys=[metadata],
            )
        ],
    )

    assert response.members[0].api_keys[0] == metadata


def test_owner_managed_api_key_create_and_capabilities_schemas():
    request = V1CreateApiKeyRequest(name="Production")
    response = V1CreateOrganizationApiKeyResponse(
        api_key="pk_secret",
        id="9ea5c01e-fac8-4aaf-9137-d2954afac230",
        name=request.name,
        preview="pk_sec…cret",
        created_at="2026-07-20T12:00:00Z",
        member_id="d3297401-bb9b-48c0-ae5e-d481159e6d4e",
        organization_id="1ad448f6-aa55-455f-a369-c26070b3fe80",
    )
    capabilities = V1ApiKeyCapabilitiesResponse(
        owner_managed_member_keys=True,
        version=3,
    )

    assert response.member_id == "d3297401-bb9b-48c0-ae5e-d481159e6d4e"
    assert capabilities.owner_managed_member_keys is True
    assert UserInfo(api_key="existing-key").api_key == "existing-key"


def test_api_call_history_accepts_specific_api_key_filter():
    request = V1ApiCallHistoryRequest(api_key="pk_specific")

    assert request.model_dump(exclude_none=True)["api_key"] == "pk_specific"
