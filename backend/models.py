from pydantic import BaseModel

class TestsuiteRequest(BaseModel):
    """Request model for test suite generation."""
    path: str

class CoverageRequest(BaseModel):
    """Request model for coverage calculation."""
    tests_path: str
    source_path: str