import pytest
from pydantic import ValidationError

from pearch import (
    OrganizationMember,
    OrganizationPendingInvite,
    OrganizationRole,
    V1InviteOrganizationMemberRequest,
    V1InviteOrganizationMemberResponse,
    V1OrganizationMembersResponse,
    V1UpdateOrganizationMemberRoleRequest,
    V1UpdateOrganizationMemberRoleResponse,
)


def test_organization_members_response_schema():
    response = V1OrganizationMembersResponse(
        organization_id="1ad448f6-aa55-455f-a369-c26070b3fe80",
        members=[
            OrganizationMember(
                user_id="d3297401-bb9b-48c0-ae5e-d481159e6d4e",
                email="member@example.com",
                name="Member Name",
                role=OrganizationRole.MEMBER,
                invited_by="4b8755c7-0688-46db-8175-cfbfa5dff5a5",
                created_at="2026-07-20T12:00:00Z",
            )
        ],
        pending_invites=[
            OrganizationPendingInvite(
                id="de2b3790-ddd7-4b0f-bf60-c34217c26375",
                email="invitee@example.com",
                role=OrganizationRole.VIEWER,
                expires_at="2026-07-27T12:00:00Z",
                invited_by="4b8755c7-0688-46db-8175-cfbfa5dff5a5",
            )
        ],
    )

    assert response.members[0].role == OrganizationRole.MEMBER
    assert response.pending_invites[0].role == OrganizationRole.VIEWER


def test_invite_organization_member_request_defaults_and_constraints():
    request = V1InviteOrganizationMemberRequest(email="invitee@example.com")

    assert request.model_dump(mode="json") == {
        "email": "invitee@example.com",
        "role": "member",
        "expires_in_days": 7,
    }

    for expires_in_days in (0, 31):
        with pytest.raises(ValidationError):
            V1InviteOrganizationMemberRequest(
                email="invitee@example.com",
                expires_in_days=expires_in_days,
            )

    with pytest.raises(ValidationError):
        V1InviteOrganizationMemberRequest(
            email="invitee@example.com",
            role="invalid",
        )


def test_invite_organization_member_response_schema():
    response = V1InviteOrganizationMemberResponse(
        invite_id="de2b3790-ddd7-4b0f-bf60-c34217c26375",
        email="invitee@example.com",
        role="admin",
        expires_at="2026-07-27T12:00:00Z",
        invite_token="invite-secret",
        invite_url="https://app.pearch.ai/invite/invite-secret",
    )

    assert response.role == OrganizationRole.ADMIN
    assert response.invite_token == "invite-secret"


def test_update_organization_member_role_schemas():
    request = V1UpdateOrganizationMemberRoleRequest(role="billing")
    response = V1UpdateOrganizationMemberRoleResponse(
        organization_id="1ad448f6-aa55-455f-a369-c26070b3fe80",
        user_id="d3297401-bb9b-48c0-ae5e-d481159e6d4e",
        updated=True,
        previous_role="member",
        role=request.role,
    )

    assert response.model_dump(mode="json")["previous_role"] == "member"
    assert response.model_dump(mode="json")["role"] == "billing"

    with pytest.raises(ValidationError):
        V1UpdateOrganizationMemberRoleRequest(role="invalid")
