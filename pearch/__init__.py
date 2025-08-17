"""Pearch client library for accessing the Pearch API."""

from .client import (
    AsyncPearchClient,
    PearchClient,
    PearchAPIError,
    PearchAuthenticationError,
    PearchValidationError,
)
from .schema import (
    V1FindMatchingJobsRequest,
    V1ProfileRequest,
    V1SearchRequest,
    V1UpsertJobsRequest,
    V2SearchRequest,
    V2SearchCompanyLeadsRequest,
    Job,
)

__version__ = "0.1.0"
__all__ = [
    "AsyncPearchClient",
    "PearchClient",
    "PearchAPIError",
    "PearchAuthenticationError",
    "PearchValidationError",
    "V1FindMatchingJobsRequest",
    "V1ProfileRequest",
    "V1SearchRequest",
    "V1UpsertJobsRequest",
    "V2SearchRequest",
    "V2SearchCompanyLeadsRequest",
    "Job",
]
