#!/usr/bin/env python
"""
Run tests for the WebUIWrapper integration with the tree search approach.
"""

import unittest
import os
import sys
import argparse

def run_tests(integration_only=False, unit_only=False, verbose=1):
    """
    Run WebUI integration tests.
    
    Args:
        integration_only: Only run integration tests
        unit_only: Only run unit tests
        verbose: Verbosity level
    """
    # Discover and load tests
    loader = unittest.TestLoader()
    
    if integration_only:
        # Only run integration tests
        suite = loader.discover('tests/tree', pattern='test_webui_integration.py')
    elif unit_only:
        # Only run unit tests
        suite = loader.discover('tests/tree', pattern='test_qua.py')
    else:
        # Run all tests
        suite = loader.discover('tests/tree', pattern='test_*.py')
    
    # Run tests
    runner = unittest.TextTestRunner(verbosity=verbose)
    result = runner.run(suite)
    
    # Return appropriate exit code
    return 0 if result.wasSuccessful() else 1

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Run WebUIWrapper tests')
    parser.add_argument('--integration-only', action='store_true', help='Only run integration tests')
    parser.add_argument('--unit-only', action='store_true', help='Only run unit tests')
    parser.add_argument('-v', '--verbose', action='count', default=1, help='Increase verbosity')
    
    args = parser.parse_args()
    
    # Check for API key if running integration tests
    if args.integration_only and not os.environ.get("WEBUI_API_KEY"):
        print("Warning: WEBUI_API_KEY environment variable not set. Integration tests may be skipped.")
    
    # Run tests
    sys.exit(run_tests(
        integration_only=args.integration_only,
        unit_only=args.unit_only,
        verbose=args.verbose
    )) 