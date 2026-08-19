import logging
from datetime import date as Date
from enum import Enum
from typing import Any, Dict, List, Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator

logger = logging.getLogger("schema.py")


class CustomFiltersMode(str, Enum):
    """
    Mode for handling custom filters in v2/search.
    
    - EXACT: Only use the passed custom_filters, no LLM-generated filters
    - SMART: Merge custom_filters with LLM-generated filters
    """
    EXACT = "exact"
    SMART = "smart"


class InsightItem(str, Enum):
    OVERALL_SUMMARY = "overall_summary"
    SHORT_QUOTES = "short_quotes"
    RATIONALE = "rationale"
    SHORT_RATIONALE = "short_rationale"


class OrganizationRole(str, Enum):
    OWNER = "owner"
    ADMIN = "admin"
    BILLING = "billing"
    MEMBER = "member"
    VIEWER = "viewer"


class FundingRound(BaseModel):
    value_usd: int | None = None
    date: Date | None = None
    round_name: str | None = None


class Language(BaseModel):
    language: str | None = None
    proficiency: str | None = None
    model_config = ConfigDict(extra="ignore")


class CompanyInfo(BaseModel):
    name: str | None = None
    domain: str | None = None
    website: str | None = None
    linkedin_url: str | None = None
    linkedin_slug: str | None = None
    crunchbase_url: str | None = None
    social_urls: List[str] | None = None
    phone_number: str | None = None
    short_address: str | None = None
    locations: List[str] | None = None

    type: str | None = None
    description: str | None = None
    industries: List[str] | None = None
    specialties: List[str] | None = None
    keywords: List[str] | None = None
    technologies: List[str] | None = None

    founded_in: int | None = None
    num_employees: int | None = None
    num_employees_range: str | None = None
    annual_revenue: int | None = None
    followers_count: int | None = None

    funding_total_usd: int | None = None
    latest_funding_amount: int | None = None
    latest_funding_round: str | None = None
    last_funding_round_date: Date | None = None
    last_funding_round_year: int | None = None
    valuation: int | None = None
    funding_rounds: List[FundingRound] | None = Field(default_factory=list)

    ipo_date: Date | None = None
    has_ipo: bool | None = None
    is_b2b: bool | None = None
    is_b2c: bool | None = None
    is_saas: bool | None = None
    is_health_tech: bool | None = None
    is_hiring: bool | None = None
    backed_by_y_combinator: bool | None = None

    icon: str | None = None
    website_score: int | None = None
    emails: List[str] | None = None
    phone_numbers: List[str] | None = None

    model_config = ConfigDict(extra="ignore")


class Experience(BaseModel):
    sequenceNo: int | None = None
    company: str | None = None
    company_domain: str | None = None
    title: str | None = None
    start_date: Date | None = None
    end_date: Date | None = None
    duration_years: float | None = None
    age_years: float | None = None
    location: str | None = None
    location_info: Dict[str, Any] | None = None
    experience_summary: str | None = None
    is_current_experience: bool | None = None
    is_startup_experience: bool | None = None
    company_info: CompanyInfo | None = None
    model_config = ConfigDict(extra="ignore")


class Education(BaseModel):
    sequenceNo: int | None = None
    university_linkedin_url: str | None = None
    campus: str | None = None
    specialization: str | None = None
    specialization_category: str | None = None
    degree: List[str] | None = None
    major: str | None = None
    start_date: Date | None = None
    end_date: Date | None = None
    model_config = ConfigDict(extra="ignore")


class Certification(BaseModel):
    title: str | None = None
    model_config = ConfigDict(extra="ignore")


class Patent(BaseModel):
    title: str | None = None
    reference: str | None = None
    description: str | None = None
    date: Date | None = None
    url: str | None = None
    model_config = ConfigDict(extra="ignore")


class CompanyExperienceGroup(BaseModel):
    company_info: CompanyInfo
    company_roles: List[Experience] = Field(default_factory=list)
    model_config = ConfigDict(extra="ignore")


class Profile(BaseModel):
    docid: str | None = None
    linkedin_slug: str | None = None
    first_name: str | None = None
    middle_name: str | None = None
    last_name: str | None = None
    picture_url: str | None = None
    title: str | None = None
    summary: str | None = None
    gender: Literal["male", "female"] | None = None
    is_decision_maker: float | None = None
    languages: List[Language] | None = Field(default_factory=list)
    inferred_languages: List[Language] | None = Field(default_factory=list)
    location: str | None = None
    is_top_universities: bool | None = None
    is_opentowork: bool | None = None
    is_hiring: bool | None = None
    total_experience_years: float | None = None
    estimated_age: float | None = None
    expertise: List[str] | None = Field(default_factory=list)
    emails: List[str] | None = Field(default_factory=list)
    best_personal_email: str | None = None
    best_business_email: str | None = None
    personal_emails: List[str] | None = Field(default_factory=list)
    business_emails: List[str] | None = Field(default_factory=list)
    has_phone_numbers: bool | None = None
    has_emails: bool | None = None
    phone_numbers: List[str] | None = Field(default_factory=list)
    phone_types: List[str] | None = Field(default_factory=list)
    websites: List[str] | None = Field(default_factory=list)
    followers_count: int | None = None
    connections_count: int | None = None

    experiences: List[CompanyExperienceGroup] | None = Field(default_factory=list)
    educations: List[Education] | None = Field(default_factory=list)
    awards: List[str] | None = Field(default_factory=list)
    certifications: List[Certification] | None = Field(default_factory=list)
    memberships: List[Any] | None = Field(default_factory=list)
    patents: List[Patent] | None = Field(default_factory=list)
    publications: List[Any] | None = Field(default_factory=list)

    outreach_message: str | None = None

    model_config = ConfigDict(extra="ignore")

    def get_all_emails(self) -> List[str]:
        """
        Get all email addresses from the profile.
        
        Returns:
            List of unique email addresses from all email fields
        """
        all_emails = []
        
        # Add from main emails list
        if self.emails:
            all_emails.extend(self.emails)
        
        # Add best emails if not None
        if self.best_personal_email:
            all_emails.append(self.best_personal_email)
        if self.best_business_email:
            all_emails.append(self.best_business_email)
        
        # Add from personal emails
        if self.personal_emails:
            all_emails.extend(self.personal_emails)
        
        # Add from business emails
        if self.business_emails:
            all_emails.extend(self.business_emails)
        
        # Return unique emails while preserving order
        seen = set()
        unique_emails = []
        for email in all_emails:
            if email and email not in seen:
                seen.add(email)
                unique_emails.append(email)
        
        return unique_emails


    def all_phone_numbers(self) -> List[str]:
        return [phone for phone in self.phone_numbers if phone]
    

class QueryInsight(BaseModel):
    match_level: str | None = None
    priority: str | None = None
    short_rationale: str | None = None
    rationale: str | None = None
    subquery: str | None = None
    short_quotes: List[str] | None = Field(default_factory=list)
    model_config = ConfigDict(extra="ignore")


class Insights(BaseModel):
    overall_summary: str | None = None
    query_insights: List[QueryInsight] | None = Field(default_factory=list)
    model_config = ConfigDict(extra="ignore")


class ScoredProfile(BaseModel):
    docid: str
    profile: Profile | None = None
    insights: Insights | None = None
    score: int | None = None
    credits_charged: int | None = None
    outreach_message: str | None = None
    model_config = ConfigDict(extra="ignore")


class RequirementMatchStats(BaseModel):
    full: float | None = None
    partial: float | None = None
    no: float | None = None
    model_config = ConfigDict(extra="ignore")


class SearchRequirementStats(BaseModel):
    requirement: str | None = None
    must_have: bool = False
    match_stats: RequirementMatchStats | None = None
    model_config = ConfigDict(extra="ignore")


class CompanyLeadResult(BaseModel):
    company: CompanyInfo | None = None
    leads: List[ScoredProfile] | None = Field(default_factory=list)
    score: float | None = None
    model_config = ConfigDict(extra="ignore")


class ExecutionDetail(BaseModel):
    status: str | None = None
    timestamp: int | None = None
    model_config = ConfigDict(extra="ignore")


class JobInsights(BaseModel):
    query_insights: List[QueryInsight] | None = Field(default_factory=list)
    model_config = ConfigDict(extra="ignore")


class JobMatch(BaseModel):
    job_id: str | None = None
    job_description: str | None = None
    score: float | None = None
    insights: JobInsights | None = None
    model_config = ConfigDict(extra="ignore")


class V2SearchResponse(BaseModel):
    uuid: str | None = None
    thread_id: str | None = None
    query: str | None = None
    user: str | None = None
    created_at: float | None = None
    duration: float | None = None
    status: str | None = None
    total_estimate: int | None = None
    total_estimate_is_lower_bound: bool | None = None
    credits_remaining: int | None = None
    credits_used: int | None = None
    credits_used_total: int | None = None
    search_results: List[ScoredProfile] | None = Field(default_factory=list)
    search_requirements_stats: List[SearchRequirementStats] | None = Field(default_factory=list)
    model_config = ConfigDict(extra="ignore")


class V2SearchCompanyLeadsResponse(BaseModel):
    uuid: str | None = None
    thread_id: str | None = None
    query: str | None = None
    search_results: List[CompanyLeadResult] | None = Field(default_factory=list)
    created_at: float | None = None
    duration: float | None = None
    user: str | None = None
    status: str | None = None
    total_estimate: int | None = None
    total_estimate_is_lower_bound: bool | None = None
    credits_remaining: int | None = None
    credits_used: int | None = None
    model_config = ConfigDict(extra="ignore")


class V1ProSearchResponse(BaseModel):
    uuid: str | None = None
    query: str | None = None
    user: str | None = None
    created_at: int | None = None
    status: str | None = None
    total_results: int | None = None
    progress: float | None = None
    credits_remaining: int | None = None
    credits_used: int | None = None
    execution_details: List[ExecutionDetail] | None = Field(default_factory=list)
    search_results: List[ScoredProfile] | None = Field(default_factory=list)
    model_config = ConfigDict(extra="ignore")


class V1UpsertJobsResponse(BaseModel):
    uuid: str | None = None
    status: str | None = None
    processed_count: int | None = None
    credits_remaining: int | None = None
    credits_used: int | None = None
    model_config = ConfigDict(extra="ignore")


class V1FindMatchingJobsResponse(BaseModel):
    uuid: str | None = None
    jobs: List[JobMatch] | None = Field(default_factory=list)
    credits_remaining: int | None = None
    credits_used: int | None = None
    model_config = ConfigDict(extra="ignore")


class V1ProfileResponse(BaseModel):
    uuid: str | None = None
    profile: Profile | None = None
    credits_remaining: int | None = None
    credits_used: int | None = None
    model_config = ConfigDict(extra="ignore")


class V1ProfileBatchError(BaseModel):
    status: str
    model_config = ConfigDict(extra="ignore")


class V1ProfileBatchItemResponse(BaseModel):
    reference_id: str | None = None
    input: Dict[str, str]
    http: int
    profile: Profile | None = None
    error: V1ProfileBatchError | None = None
    model_config = ConfigDict(extra="ignore")


class V1ProfileBatchResponse(BaseModel):
    uuid: str
    results: List[V1ProfileBatchItemResponse] = Field(default_factory=list)
    profiles_succeeded: int
    profiles_failed: int
    credits_remaining: int | None = None
    credits_used: int = 0
    credits_breakdown: Dict[str, Any] | None = None
    model_config = ConfigDict(extra="ignore")


# Request parameter classes for each endpoint


class CustomFilters(BaseModel):
    locations: List[str] | None = None
    languages: List[str] | None = None
    titles: List[str] | None = None
    industries: List[str] | None = None
    companies: List[str] | None = None
    universities: List[str] | None = None
    keywords: List[str] | None = None
    min_linkedin_followers: int | None = None
    max_linkedin_followers: int | None = None
    min_total_experience_years: float | None = None
    max_total_experience_years: float | None = None
    min_estimated_age: float | None = None
    max_estimated_age: float | None = None
    studied_at_top_universities: bool | None = None
    first_name: str | None = None
    middle_name: str | None = None
    last_name: str | None = None
    gender: Literal["male", "female"] | None = None
    has_startup_experience: bool | None = None
    has_saas_experience: bool | None = None
    has_b2b_experience: bool | None = None
    has_b2c_experience: bool | None = None
    min_current_experience_years: float | None = None
    max_current_experience_years: float | None = None
    degrees: List[Literal["bachelor", "master", "MBA", "doctor", "postdoc"]] | None = None
    specialization_categories: List[Literal[
        "Business & Management",
        "Finance",
        "Engineering",
        "Computer Science & IT",
        "Health & Medicine",
        "Social Sciences",
        "Natural Sciences",
        "Mathematics",
        "Education",
        "Communication & Media",
        "Arts, Design & Architecture",
        "Humanities & Liberal Arts",
        "Law & Criminal Justice"
    ]] | None = None
    model_config = ConfigDict(extra="forbid")


class SearchRequirement(BaseModel):
    search_requirement: str
    must_have: bool = False
    model_config = ConfigDict(extra="forbid")


class V2SearchRequest(BaseModel):
    query: str | None = None
    search_requirements: List[SearchRequirement] | None = None
    thread_id: str | None = None
    type: Literal["superfast", "fast", "pro"] | None = None
    time_budget: int | None = Field(default=None, gt=0)
    insights: bool | None = None
    insights_items: List[InsightItem] | None = None
    high_freshness: bool | None = None
    profile_scoring: bool | None = None
    fill_with_low_confidence_results: bool = Field(
        default=False,
        exclude_if=lambda value: value is False,
    )
    custom_filters: CustomFilters | None = None
    custom_filters_mode: CustomFiltersMode | None = None
    strict_filters: bool | None = None
    filter_out_no_emails: bool | None = None
    reveal_emails: bool | None = None
    filter_out_no_phones: bool | None = None
    filter_out_no_phones_or_emails: bool | None = None
    reveal_phones: bool | None = None
    limit: int | None = Field(default=None, ge=1, le=1000)
    offset: int | None = Field(default=None, ge=0)
    docid_blacklist: List[str] | None = None
    docid_whitelist: List[str] | None = None
    require_emails: bool | None = None
    show_emails: bool | None = None
    require_phone_numbers: bool | None = None
    require_phones_or_emails: bool | None = None
    show_phone_numbers: bool | None = None
    free: bool | None = None
    async_: bool | None = Field(default=None, alias="async")
    model_config = ConfigDict(extra="ignore", populate_by_name=True)


class V2SearchCountRequest(BaseModel):
    """Count profiles using natural-language search, structured requirements, or filters.

    A filter-only request costs 2 credits. A request containing ``query`` or
    ``search_requirements`` costs 5 credits. Setting ``stats=True`` adds 5 credits.
    """

    query: str | None = Field(
        default=None,
        description="Natural-language people-search query. A query-based count costs 5 credits.",
    )
    search_requirements: List[SearchRequirement] | None = Field(
        default=None,
        description="Structured search requirements. A requirements-based count costs 5 credits.",
    )
    strict_filters: bool | None = None
    custom_filters: CustomFilters | None = None
    custom_filters_mode: CustomFiltersMode | None = None
    docid_blacklist: List[str] | None = None
    docid_whitelist: List[str] | None = None
    exclude_liked_profiles: bool | None = None
    enable_custom_search: bool | None = None
    stats: bool | None = Field(
        default=None,
        description="Return statistics from up to 1,000 matching profiles for an additional 5 credits.",
    )
    model_config = ConfigDict(extra="ignore")


class V2SearchCountTopValue(BaseModel):
    value: str
    profile_count: int
    percentage: float
    model_config = ConfigDict(extra="ignore")


class V2SearchCountBucket(BaseModel):
    label: str
    profile_count: int
    percentage: float
    model_config = ConfigDict(extra="ignore")


class V2SearchCountFieldStats(BaseModel):
    profiles_total: int
    profiles_non_empty: int | None = None
    profiles_empty: int | None = None
    coverage_pct: float
    values_total: int | None = None
    unique_values: int | None = None
    top_values: List[V2SearchCountTopValue] | None = None
    min: float | None = None
    max: float | None = None
    mean: float | None = None
    median: float | None = None
    buckets: List[V2SearchCountBucket] | None = None
    counts: Dict[str, int] | None = None
    model_config = ConfigDict(extra="ignore")


class V2SearchCountStats(BaseModel):
    matched_profiles: int
    sample_limit: int
    top_values_limit: int
    sample_size: int
    truncated: bool
    fields: Dict[str, V2SearchCountFieldStats] = Field(default_factory=dict)
    model_config = ConfigDict(extra="ignore")


class V2SearchCountResponse(BaseModel):
    count: int
    approximate: bool | None = None
    uuid: str | None = None
    credits_used: int | None = None
    credits_remaining: int | None = None
    stats: V2SearchCountStats | None = None
    model_config = ConfigDict(extra="ignore")


class V2SearchCompanyLeadsRequest(BaseModel):
    company_query: str
    thread_id: str | None = None
    lead_query: str | None = None
    outreach_message_instruction: str | None = None
    limit: int | None = Field(default=50, ge=1, le=1000)
    leads_limit: int | None = Field(default=3, ge=1, le=10)
    reveal_emails: bool | None = False
    reveal_phones: bool | None = False
    filter_out_no_emails: bool | None = False
    filter_out_no_phones: bool | None = False
    filter_out_no_phones_or_emails: bool | None = False
    high_freshness: bool | None = False
    company_high_freshness: bool | None = False
    select_top_leads: bool | None = True
    show_emails: bool | None = None
    show_phone_numbers: bool | None = None
    require_emails: bool | None = None
    require_phone_numbers: bool | None = None
    require_phones_or_emails: bool | None = None
    model_config = ConfigDict(extra="ignore")


class V2CompanyRequest(BaseModel):
    linkedin_slug: str | None = Field(default=None, min_length=1, max_length=512)
    domain: str | None = Field(default=None, min_length=1, max_length=512)
    model_config = ConfigDict(extra="forbid")

    @model_validator(mode="after")
    def require_one_company_identifier(self):
        if bool(self.linkedin_slug) == bool(self.domain):
            raise ValueError("Provide exactly one of 'linkedin_slug' or 'domain'.")
        return self


class V2CompanyResponse(BaseModel):
    company: CompanyInfo
    credits_remaining: int | None = None
    credits_used: int | None = None
    credits_breakdown: Dict[str, Any] | None = None
    model_config = ConfigDict(extra="ignore")


class V1SearchRequest(BaseModel):
    query: str | None = None
    type: str | None = None
    filter_out_no_emails: bool | None = None
    require_emails: bool | None = None
    limit: int | None = None
    model_config = ConfigDict(extra="ignore")


class Job(BaseModel):
    job_id: str
    job_description: str
    model_config = ConfigDict(extra="ignore")


class ListedJob(BaseModel):
    job_id: str
    job_description: str
    created_at: str | None = None
    model_config = ConfigDict(extra="ignore")


class V1UpsertJobsRequest(BaseModel):
    jobs: List[Job]
    replace: bool | None = False
    model_config = ConfigDict(extra="ignore")


class V1FindMatchingJobsRequest(BaseModel):
    profile: Dict[str, Any]
    limit: int | None = Field(default=10, ge=1, le=100)
    model_config = ConfigDict(extra="ignore")


class V1ProfileRequest(BaseModel):
    docid: str
    high_freshness: bool | None = False
    reveal_emails: bool | None = False
    reveal_phones: bool | None = False
    with_profile: bool | None = False
    show_emails: bool | None = None
    show_phone_numbers: bool | None = None
    model_config = ConfigDict(extra="ignore")


class V1ProfileBatchItem(BaseModel):
    reference_id: str | None = Field(default=None, min_length=1, max_length=512)
    docid: str | None = Field(default=None, min_length=1, max_length=512)
    uuid: str | None = Field(default=None, min_length=1, max_length=512)
    email: str | None = Field(default=None, min_length=1, max_length=512)
    model_config = ConfigDict(extra="forbid")

    @model_validator(mode="after")
    def require_one_profile_identifier(self):
        if sum(bool(value) for value in (self.docid, self.uuid, self.email)) != 1:
            raise ValueError("Provide exactly one of 'docid', 'uuid' or 'email'.")
        return self


class V1ProfileBatchRequest(BaseModel):
    profiles: List[V1ProfileBatchItem] = Field(min_length=1, max_length=100)
    high_freshness: bool | None = False
    reveal_emails: bool | None = False
    reveal_phones: bool | None = False
    with_profile: bool | None = False
    github_enrich: bool | None = None
    short_response: bool = False
    omit_fields: List[str] | None = None
    show_emails: bool | None = None
    show_phone_numbers: bool | None = None
    model_config = ConfigDict(extra="forbid")

    @model_validator(mode="after")
    def reject_duplicate_identifiers(self):
        seen = set()
        for item in self.profiles:
            if item.docid:
                key = ("docid", item.docid.lower())
            elif item.uuid:
                key = ("uuid", item.uuid.lower())
            else:
                key = ("email", str(item.email).lower())
            if key in seen:
                raise ValueError(f"Duplicate profile identifier: '{key[0]}={key[1]}'.")
            seen.add(key)
        return self


class V2SearchSubmitResponse(BaseModel):
    task_id: str
    thread_id: str | None = None
    status: str
    message: str
    model_config = ConfigDict(extra="ignore")


class V2SearchStatusResponse(BaseModel):
    task_id: str
    status: str | None = None
    created_at: str | None = None
    query: str | None = None
    result: V2SearchResponse | None = None
    duration: float | None = None
    error: str | None = None
    started_at: str | None = None
    credits_used: int | None = None
    credits_remaining: int | None = None
    model_config = ConfigDict(extra="ignore")


class ApiCallHistoryEntry(BaseModel):
    uuid: str | None = None
    thread_id: str | None = None
    path: str | None = None
    parameters: Dict[str, Any] | None = None
    items_count: int | None = None
    created_at: str | None = None
    response_time: float | None = None
    response_status: int | None = None
    error_message: str | None = None
    task_status: str | None = None
    credits_used: int | None = None
    model_config = ConfigDict(extra="ignore")


class V1ApiCallHistoryRequest(BaseModel):
    limit: int | None = Field(default=10, ge=1, le=1000)
    paths: List[str] | None = None
    api_key: str | None = None
    model_config = ConfigDict(extra="ignore")


class V1ApiCallHistoryResponse(BaseModel):
    api_call_history: List[ApiCallHistoryEntry] | None = Field(default_factory=list)
    user: str | None = None
    total_credits_used: int | None = None
    model_config = ConfigDict(extra="ignore")


class ApiKeyMetadata(BaseModel):
    id: str
    name: str
    preview: str
    created_at: str
    last_used_at: str | None = None
    revoked_at: str | None = None
    model_config = ConfigDict(extra="ignore")


class OrganizationMemberApiKeys(BaseModel):
    user_id: str
    email: str | None = None
    role: str
    api_keys: List[ApiKeyMetadata] = Field(default_factory=list)
    model_config = ConfigDict(extra="ignore")


class V1ApiKeyCapabilitiesResponse(BaseModel):
    owner_managed_member_keys: bool
    version: int
    model_config = ConfigDict(extra="ignore")


class V1OrganizationApiKeysResponse(BaseModel):
    organization_id: str
    members: List[OrganizationMemberApiKeys] = Field(default_factory=list)
    model_config = ConfigDict(extra="ignore")


class V1CreateApiKeyRequest(BaseModel):
    name: str = Field(min_length=1, max_length=100)
    model_config = ConfigDict(extra="forbid")


class V1CreateOrganizationApiKeyResponse(BaseModel):
    api_key: str
    id: str
    name: str
    preview: str
    created_at: str
    member_user_id: str
    organization_id: str
    model_config = ConfigDict(extra="ignore")


class OrganizationMember(BaseModel):
    user_id: str
    email: str
    name: str | None = None
    role: OrganizationRole
    invited_by: str | None = None
    created_at: str
    model_config = ConfigDict(extra="ignore")


class OrganizationPendingInvite(BaseModel):
    id: str
    email: str
    role: OrganizationRole
    expires_at: str
    invited_by: str | None = None
    model_config = ConfigDict(extra="ignore")


class V1OrganizationMembersResponse(BaseModel):
    organization_id: str
    members: List[OrganizationMember] = Field(default_factory=list)
    pending_invites: List[OrganizationPendingInvite] = Field(default_factory=list)
    model_config = ConfigDict(extra="ignore")


class V1InviteOrganizationMemberRequest(BaseModel):
    email: str
    role: OrganizationRole = OrganizationRole.MEMBER
    expires_in_days: int = Field(default=7, ge=1, le=30)
    model_config = ConfigDict(extra="forbid")


class V1InviteOrganizationMemberResponse(BaseModel):
    invite_id: str
    email: str
    role: OrganizationRole
    expires_at: str
    invite_token: str
    invite_url: str
    model_config = ConfigDict(extra="ignore")


class V1UpdateOrganizationMemberRoleRequest(BaseModel):
    role: OrganizationRole
    model_config = ConfigDict(extra="forbid")


class V1UpdateOrganizationMemberRoleResponse(BaseModel):
    organization_id: str
    user_id: str
    updated: bool
    previous_role: OrganizationRole
    role: OrganizationRole
    model_config = ConfigDict(extra="ignore")


class UserInfo(BaseModel):
    api_key: str | None = None
    email: str | None = None
    sub_client: Any | None = None
    model_config = ConfigDict(extra="ignore")


class PricingInfo(BaseModel):
    id: str
    credits: float | int
    description: str | None = None
    model_config = ConfigDict(extra="ignore")


class V1ListJobsRequest(BaseModel):
    limit: int | None = Field(default=100, ge=1, le=1000)
    model_config = ConfigDict(extra="ignore")


class V1ListJobsResponse(BaseModel):
    status: str | None = None
    jobs: List[ListedJob] | None = Field(default_factory=list)
    total_count: int | None = None
    returned_count: int | None = None
    model_config = ConfigDict(extra="ignore")


class V1DeleteJobsRequest(BaseModel):
    job_ids: List[str]
    model_config = ConfigDict(extra="ignore")


class V1DeleteJobsResponse(BaseModel):
    uuid: str | None = None
    status: str | None = None
    deleted_count: int | None = None
    errors: List[str] | None = Field(default_factory=list)
    model_config = ConfigDict(extra="ignore")


class V1UserResponse(BaseModel):
    user: UserInfo | None = None
    credits_remaining: int | None = None
    pricing: List[PricingInfo] | None = Field(default_factory=list)
    model_config = ConfigDict(extra="ignore")
