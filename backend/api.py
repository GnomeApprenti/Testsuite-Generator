import traceback
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware

from models import TestsuiteRequest, CoverageRequest
from generator import generator
from tools import CoverageCalculatorTool

def create_app() -> FastAPI:
    """Create and configure the FastAPI application."""
    app = FastAPI(
        title="Test Suite Generator API",
        description="API for generating test suites using AI agents",
        version="1.0.0"
    )

    # Add CORS middleware
    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    @app.post("/generate_testsuite")
    async def run_testsuite_generator(request: TestsuiteRequest):
        """
        Generate a test suite for the given project path.
        
        Args:
            request: Contains the path to the project directory
            
        Returns:
            dict: Contains the generated test suite code
        """
        try:
            print(f"Received project path: {request.path}")
            result = generator.generate(request.path)
            return {"testsuite": result}
        except Exception as e:
            traceback.print_exc()
            raise HTTPException(status_code=500, detail=str(e))

    @app.post("/calculate_coverage")
    async def calculate_coverage(req: CoverageRequest):
        """
        Calculate code coverage for the given test and source paths.
        
        Args:
            req: Contains paths to tests and source code
            
        Returns:
            dict: Contains the coverage report
        """
        try:
            tool = CoverageCalculatorTool()
            result = tool.forward(req.tests_path, req.source_path)
            return {"coverage_report": result}
        except Exception as e:
            traceback.print_exc()
            raise HTTPException(status_code=500, detail=str(e))

    @app.get("/health")
    async def health_check():
        """Health check endpoint."""
        return {"status": "healthy"}

    return app