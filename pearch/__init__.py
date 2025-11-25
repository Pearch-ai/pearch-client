"""Pearch client library for accessing the Pearch API."""

from .client import (
    AsyncPearchClient,
    PearchClient,
    PearchAPIError,
    PearchAuthenticationError,
    PearchValidationError,
)
from .schema import (
    CustomFilters,
    V1FindMatchingJobsRequest,
    V1ProfileRequest,
    V1SearchRequest,
    V1UpsertJobsRequest,
    V1ListJobsRequest,
    V1DeleteJobsRequest,
    V2SearchRequest,
    V2SearchCompanyLeadsRequest,
    Job,
    ListedJob,
)

__version__ = "0.1.0"
__all__ = [
    "AsyncPearchClient",
    "PearchClient",
    "PearchAPIError",
    "PearchAuthenticationError",
    "PearchValidationError",
    "CustomFilters",
    "V1FindMatchingJobsRequest",
    "V1ProfileRequest",
    "V1SearchRequest",
    "V1UpsertJobsRequest",
    "V1ListJobsRequest",
    "V1DeleteJobsRequest",
    "V2SearchRequest",
    "V2SearchCompanyLeadsRequest",
    "Job",
    "ListedJob",
]
