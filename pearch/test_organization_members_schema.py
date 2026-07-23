import pytest
from pydantic import ValidationError

from pearch import (
    OrganizationMember,
    OrganizationRole,
    V1AddOrganizationMemberRequest,
    V1AddOrganizationMemberResponse,
    V1OrganizationMembersResponse,
    V1UpdateOrganizationMemberRoleRequest,
    V1UpdateOrganizationMemberRoleResponse,
)


def test_organization_members_response_schema():
    response = V1OrganizationMembersResponse(
        organization_id="1ad448f6-aa55-455f-a369-c26070b3fe80",
        members=[
            OrganizationMember(
                id="d3297401-bb9b-48c0-ae5e-d481159e6d4e",
                user_id=None,
                email="member@example.com",
                name=None,
                role=OrganizationRole.MEMBER,
                added_by="4b8755c7-0688-46db-8175-cfbfa5dff5a5",
                created_at="2026-07-20T12:00:00Z",
            )
        ],
    )

    assert response.members[0].role == OrganizationRole.MEMBER
    assert response.members[0].added_by == "4b8755c7-0688-46db-8175-cfbfa5dff5a5"
    assert "pending_invites" not in response.model_dump()


def test_add_organization_member_request_defaults_and_constraints():
    request = V1AddOrganizationMemberRequest(email="member@example.com")

    assert request.model_dump(mode="json") == {
        "email": "member@example.com",
        "role": "member",
    }

    with pytest.raises(ValidationError):
        V1AddOrganizationMemberRequest(email="member@example.com", expires_in_days=7)

    with pytest.raises(ValidationError):
        V1AddOrganizationMemberRequest(
            email="member@example.com",
            role="invalid",
        )


def test_add_organization_member_response_schema():
    response = V1AddOrganizationMemberResponse(
        organization_id="1ad448f6-aa55-455f-a369-c26070b3fe80",
        id="d3297401-bb9b-48c0-ae5e-d481159e6d4e",
        user_id=None,
        email="member@example.com",
        name=None,
        role="admin",
        added_by="4b8755c7-0688-46db-8175-cfbfa5dff5a5",
        created_at="2026-07-20T12:00:00Z",
    )

    assert response.role == OrganizationRole.ADMIN
    assert response.id == "d3297401-bb9b-48c0-ae5e-d481159e6d4e"
    assert response.user_id is None


def test_update_organization_member_role_schemas():
    request = V1UpdateOrganizationMemberRoleRequest(role="billing")
    response = V1UpdateOrganizationMemberRoleResponse(
        organization_id="1ad448f6-aa55-455f-a369-c26070b3fe80",
        member_id="d3297401-bb9b-48c0-ae5e-d481159e6d4e",
        updated=True,
        previous_role="member",
        role=request.role,
    )

    assert response.model_dump(mode="json")["previous_role"] == "member"
    assert response.model_dump(mode="json")["role"] == "billing"

    with pytest.raises(ValidationError):
        V1UpdateOrganizationMemberRoleRequest(role="invalid")
