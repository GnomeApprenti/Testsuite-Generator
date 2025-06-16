#!/usr/bin/env python3
"""
Test Suite Generator Application

A FastAPI application that generates test suites for Python projects using AI agents.
"""

import uvicorn
from config import config
from telemetry import telemetry
from api import create_app

def main():
    """Main entry point for the application."""
    # Setup telemetry
    telemetry.setup_tracing()
    
    # Create the FastAPI app
    app = create_app()
    
    # Run the application
    uvicorn.run(app, host="0.0.0.0", port=8080)

if __name__ == "__main__":
    main()