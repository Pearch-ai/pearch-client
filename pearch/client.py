import logging
import os
from typing import Any, Dict, Optional

import httpx

from .schema import (
    Profile,
    V1FindMatchingJobsRequest,
    V1FindMatchingJobsResponse,
    V1ProfileRequest,
    V1ProfileResponse,
    V1SearchRequest,
    V1ProSearchResponse,
    V1UpsertJobsRequest,
    V1UpsertJobsResponse,
    V2SearchCompanyLeadsRequest,
    V2SearchCompanyLeadsResponse,
    V2SearchRequest,
    V2SearchResponse,
)

logging.basicConfig(
    level=logging.INFO,
    format="%(levelname)s - %(name)s - %(filename)s:%(lineno)d - %(message)s",
)

logger = logging.getLogger(__name__)


class PearchAPIError(Exception):
    """Base exception for Pearch API errors"""

    def __init__(
        self,
        message: str,
        status_code: int | None = None,
        response_data: Dict[str, Any] | None = None,
    ):
        super().__init__(message)
        self.status_code = status_code
        self.response_data = response_data or {}


class PearchAuthenticationError(PearchAPIError):
    """Raised when authentication fails"""

    pass


class PearchValidationError(PearchAPIError):
    """Raised when request validation fails"""

    pass


class PearchClient:
    """
    Synchronous client for the Pearch.AI API

    This client provides access to all Pearch.AI API endpoints with proper
    type hints, error handling, and authentication management.
    """

    def __init__(
        self,
        api_key: Optional[str] = None,
        base_url: str = None,
        timeout: float = 30.0,
        max_retries: int = 3,
        token: Optional[str] = None,
        **kwargs,
    ):
        """
        Initialize the Pearch API client

        Args:
            api_key: Your Pearch.AI API key (defaults to PEARCH_API_KEY environment variable)
            base_url: Base URL for the API (default: PEARCH_API_URL or https://api.pearch.ai/)
            timeout: Request timeout in seconds (default: 30.0)
            max_retries: Maximum number of retries for failed requests (default: 3)
            token: Optional test token (sent as X-Test-Secret header)
            **kwargs: Additional arguments passed to httpx.Client
        """
        # Use environment variable if api_key not provided
        if api_key is None:
            api_key = os.getenv("PEARCH_API_KEY")

        if base_url is None:
            base_url = os.getenv("PEARCH_API_URL") or "https://api.pearch.ai/"

        if not api_key:
            raise ValueError(
                "API key is required. Provide it as a parameter or set the PEARCH_API_KEY environment variable."
            )

        self.api_key = api_key
        self.token = token
        self.base_url = base_url.rstrip("/") + "/"

        headers = {
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json",
            "User-Agent": "pearch-python-client/1.0",
        }

        if token:
            headers["X-Test-Secret"] = token

        self._client = httpx.Client(
            base_url=self.base_url, timeout=timeout, headers=headers, **kwargs
        )

        # Configure retries
        transport = httpx.HTTPTransport(retries=max_retries)
        self._client._transport = transport

    def __enter__(self):
        """Context manager entry"""
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit"""
        self.close()

    def close(self):
        """Close the HTTP client"""
        self._client.close()

    def _make_request(
        self,
        method: str,
        endpoint: str,
        data: Optional[Dict[str, Any]] = None,
        params: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        """
        Make an HTTP request to the API

        Args:
            method: HTTP method (GET, POST, etc.)
            endpoint: API endpoint path
            data: Request body data for POST requests
            params: Query parameters for GET requests

        Returns:
            Parsed JSON response

        Raises:
            PearchAuthenticationError: When API key is invalid
            PearchValidationError: When request parameters are invalid
            PearchAPIError: For other API errors
        """
        logger.debug(f"Making {method} request to {endpoint}")

        response = self._client.request(
            method=method, url=endpoint, json=data, params=params
        )

        # Handle different HTTP status codes
        if response.status_code == 401:
            raise PearchAuthenticationError(
                "Invalid API key",
                status_code=401,
                response_data=response.json() if response.content else {},
            )
        elif response.status_code == 400:
            error_data = response.json() if response.content else {}
            raise PearchValidationError(
                f"Invalid request parameters: {error_data}",
                status_code=400,
                response_data=error_data,
            )
        elif response.status_code >= 400:
            error_data = response.json() if response.content else {}
            raise PearchAPIError(
                f"API request failed with status {response.status_code}: {error_data}",
                status_code=response.status_code,
                response_data=error_data,
            )

        response.raise_for_status()
        return response.json()

    # V2 API Methods

    def search(self, request: V2SearchRequest) -> V2SearchResponse:
        """
        Execute an enhanced search query (v2 endpoint)

        This is the recommended search endpoint with advanced parameters for
        granular control over search behavior.

        Args:
            request: Search request parameters

        Returns:
            Search response with results and metadata
        """
        logger.info(f"Executing v2 search with query: {request.query}")

        response_data = self._make_request(
            method="POST",
            endpoint="v2/search",
            data=request.model_dump(exclude_none=True),
        )

        return V2SearchResponse(**response_data)

    def search_company_leads(
        self, request: V2SearchCompanyLeadsRequest
    ) -> V2SearchCompanyLeadsResponse:
        """
        Search for companies and their leads

        Find companies based on a query and optionally find leads within those
        companies with personalized outreach messages.

        Args:
            request: Company and lead search parameters

        Returns:
            Company search response with leads
        """
        logger.info(f"Searching company leads with query: {request.company_query}")

        response_data = self._make_request(
            method="POST",
            endpoint="v2/search_company_leads",
            data=request.model_dump(exclude_none=True),
        )

        return V2SearchCompanyLeadsResponse(**response_data)

    # V1 API Methods (Legacy)

    def search_v1(self, request: V1SearchRequest) -> V1ProSearchResponse:
        """
        Execute a search query (v1 legacy endpoint)

        Legacy search endpoint that accepts parameters as query parameters.
        Consider using the v2 search endpoint for new implementations.

        Args:
            request: Search request parameters

        Returns:
            Search response with results and metadata
        """
        logger.info(f"Executing v1 search with query: {request.query}")

        response_data = self._make_request(
            method="GET",
            endpoint="v1/search",
            params=request.model_dump(exclude_none=True),
        )

        return V1ProSearchResponse(**response_data)

    def upsert_jobs(self, request: V1UpsertJobsRequest) -> V1UpsertJobsResponse:
        """
        Upload or update job listings

        Submit a list of jobs to be indexed. If job_id already exists, it will
        be updated. Supports both upsert and replace modes.

        Args:
            request: Job upsert request with job listings

        Returns:
            Upsert response with processing status
        """
        logger.info(f"Upserting {len(request.jobs)} jobs (replace={request.replace})")

        response_data = self._make_request(
            method="POST",
            endpoint="v1/upsert_jobs",
            data=request.model_dump(exclude_none=True),
        )

        return V1UpsertJobsResponse(**response_data)

    def find_matching_jobs(
        self, request: V1FindMatchingJobsRequest
    ) -> V1FindMatchingJobsResponse:
        """
        Find jobs relevant to a candidate profile

        Accepts a profile in arbitrary JSON format and returns a list of
        relevant jobs with match scores and insights.

        Args:
            request: Job matching request with candidate profile

        Returns:
            Matching jobs response with relevance scores
        """
        logger.info(f"Finding matching jobs for profile (limit={request.limit})")

        response_data = self._make_request(
            method="POST",
            endpoint="v1/find_matching_jobs",
            data=request.model_dump(exclude_none=True),
        )

        return V1FindMatchingJobsResponse(**response_data)

    def get_profile(self, request: V1ProfileRequest) -> V1ProfileResponse:
        """
        Retrieve a user profile by LinkedIn profile ID

        Get detailed profile information for a specific LinkedIn profile ID
        with optional real-time updates and contact information.

        Args:
            request: Profile request with docid and options

        Returns:
            Profile response with detailed candidate information
        """
        logger.info(f"Retrieving profile for docid: {request.docid}")

        response_data = self._make_request(
            method="GET",
            endpoint="v1/profile",
            params=request.model_dump(exclude_none=True),
        )

        return V1ProfileResponse(**response_data)


class AsyncPearchClient:
    """
    Asynchronous client for the Pearch.AI API

    This client provides async access to all Pearch.AI API endpoints with proper
    type hints, error handling, and authentication management.
    """

    def __init__(
        self,
        api_key: Optional[str] = None,
        base_url: Optional[str] = None,
        timeout: float = 30.0,
        max_retries: int = 3,
        token: Optional[str] = None,
        **kwargs,
    ):
        """
        Initialize the async Pearch API client

        Args:
            api_key: Your Pearch.AI API key (defaults to PEARCH_API_KEY environment variable)
            base_url: Base URL for the API (default: PEARCH_API_URL or https://api.pearch.ai/)
            timeout: Request timeout in seconds (default: 30.0)
            max_retries: Maximum number of retries for failed requests (default: 3)
            token: Optional test token (sent as X-Test-Secret header)
            **kwargs: Additional arguments passed to httpx.AsyncClient
        """
        # Use environment variable if api_key not provided
        if base_url is None:
            base_url = os.getenv("PEARCH_API_URL") or "https://api.pearch.ai/"

        if api_key is None:
            api_key = os.getenv("PEARCH_API_KEY")

        if not api_key:
            raise ValueError(
                "API key is required. Provide it as a parameter or set the PEARCH_API_KEY environment variable."
            )
        if token is None:
            token = os.getenv("PEARCH_TEST_KEY")

        self.api_key = api_key
        self.token = token
        self.base_url = base_url.rstrip("/") + "/"

        headers = {
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json",
            "User-Agent": "pearch-python-client/1.0",
        }

        if token:
            headers["X-Test-Secret"] = token

        self._client = httpx.AsyncClient(
            base_url=self.base_url, timeout=timeout, headers=headers, **kwargs
        )

        # Configure retries
        transport = httpx.AsyncHTTPTransport(retries=max_retries)
        self._client._transport = transport

    async def __aenter__(self):
        """Async context manager entry"""
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        """Async context manager exit"""
        await self.close()

    async def close(self):
        """Close the HTTP client"""
        await self._client.aclose()

    async def _make_request(
        self,
        method: str,
        endpoint: str,
        data: Optional[Dict[str, Any]] = None,
        params: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        """
        Make an async HTTP request to the API

        Args:
            method: HTTP method (GET, POST, etc.)
            endpoint: API endpoint path
            data: Request body data for POST requests
            params: Query parameters for GET requests

        Returns:
            Parsed JSON response

        Raises:
            PearchAuthenticationError: When API key is invalid
            PearchValidationError: When request parameters are invalid
            PearchAPIError: For other API errors
        """
        logger.debug(f"Making async {method} request to {endpoint}")

        response = await self._client.request(
            method=method, url=endpoint, json=data, params=params
        )

        # Handle different HTTP status codes
        if response.status_code == 401:
            raise PearchAuthenticationError(
                "Invalid API key",
                status_code=401,
                response_data=response.json() if response.content else {},
            )
        elif response.status_code == 400:
            error_data = response.json() if response.content else {}
            raise PearchValidationError(
                f"Invalid request parameters: {error_data}",
                status_code=400,
                response_data=error_data,
            )
        elif response.status_code >= 400:
            error_data = response.json() if response.content else {}
            raise PearchAPIError(
                f"API request failed with status {response.status_code}: {error_data}",
                status_code=response.status_code,
                response_data=error_data,
            )

        response.raise_for_status()
        return response.json()

    # V2 API Methods

    async def search(self, request: V2SearchRequest) -> V2SearchResponse:
        """
        Execute an enhanced search query (v2 endpoint) - async version

        Args:
            request: Search request parameters

        Returns:
            Search response with results and metadata
        """
        logger.info(f"Executing async v2 search with query: {request.query}")

        response_data = await self._make_request(
            method="POST",
            endpoint="v2/search",
            data=request.model_dump(exclude_none=True),
        )

        return V2SearchResponse(**response_data)

    async def search_company_leads(
        self, request: V2SearchCompanyLeadsRequest
    ) -> V2SearchCompanyLeadsResponse:
        """
        Search for companies and their leads - async version

        Args:
            request: Company and lead search parameters

        Returns:
            Company search response with leads
        """
        logger.info(
            f"Searching company leads async with query: {request.company_query}"
        )

        response_data = await self._make_request(
            method="POST",
            endpoint="v2/search_company_leads",
            data=request.model_dump(exclude_none=True),
        )

        return V2SearchCompanyLeadsResponse(**response_data)

    # V1 API Methods (Legacy)

    async def search_v1(self, request: V1SearchRequest) -> V1ProSearchResponse:
        """
        Execute a search query (v1 legacy endpoint) - async version

        Args:
            request: Search request parameters

        Returns:
            Search response with results and metadata
        """
        logger.info(f"Executing async v1 search with query: {request.query}")

        response_data = await self._make_request(
            method="GET",
            endpoint="v1/search",
            params=request.model_dump(exclude_none=True),
        )
        if request.type == "fast":
            return [Profile(**result) for result in response_data]
        else:
            return V1ProSearchResponse(**response_data)

    async def upsert_jobs(self, request: V1UpsertJobsRequest) -> V1UpsertJobsResponse:
        """
        Upload or update job listings - async version

        Args:
            request: Job upsert request with job listings

        Returns:
            Upsert response with processing status
        """
        logger.info(
            f"Upserting {len(request.jobs)} jobs async (replace={request.replace})"
        )

        response_data = await self._make_request(
            method="POST",
            endpoint="v1/upsert_jobs",
            data=request.model_dump(exclude_none=True),
        )

        return V1UpsertJobsResponse(**response_data)

    async def find_matching_jobs(
        self, request: V1FindMatchingJobsRequest
    ) -> V1FindMatchingJobsResponse:
        """
        Find jobs relevant to a candidate profile - async version

        Args:
            request: Job matching request with candidate profile

        Returns:
            Matching jobs response with relevance scores
        """
        logger.info(f"Finding matching jobs async for profile (limit={request.limit})")

        response_data = await self._make_request(
            method="POST",
            endpoint="v1/find_matching_jobs",
            data=request.model_dump(exclude_none=True),
        )

        return V1FindMatchingJobsResponse(**response_data)

    async def get_profile(self, request: V1ProfileRequest) -> V1ProfileResponse:
        """
        Retrieve a user profile by LinkedIn profile ID - async version

        Args:
            request: Profile request with docid and options

        Returns:
            Profile response with detailed candidate information
        """
        logger.info(f"Retrieving profile async for docid: {request.docid}")

        response_data = await self._make_request(
            method="GET",
            endpoint="v1/profile",
            params=request.model_dump(exclude_none=True),
        )
        return V1ProfileResponse(**response_data)
