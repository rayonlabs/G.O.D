#!/usr/bin/env python3

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from validator.tests.test_tournament_organiser import test_all_organiser_functionality
from validator.tests.test_tournament_integration import test_all_tournament_integration
from validator.utils.logging import get_logger

logger = get_logger(__name__)


def run_all_tournament_tests():
    test_results = {}
    
    logger.info("Starting tournament test suite...")
    
    logger.info("Running tournament organiser tests...")
    test_results["organiser"] = test_all_organiser_functionality()
    
    logger.info("Running tournament integration tests...")
    test_results["integration"] = test_all_tournament_integration()
    
    passed_tests = sum(test_results.values())
    total_tests = len(test_results)
    
    logger.info(f"\n{'='*50}")
    logger.info("TOURNAMENT TEST RESULTS:")
    logger.info(f"{'='*50}")
    
    for test_name, result in test_results.items():
        status = "PASSED" if result else "FAILED"
        logger.info(f"{test_name.capitalize()} tests: {status}")
    
    logger.info(f"\nOverall: {passed_tests}/{total_tests} test suites passed")
    
    if passed_tests == total_tests:
        logger.info("🎉 All tournament tests passed!")
        return True
    else:
        logger.error("❌ Some tournament tests failed!")
        return False


if __name__ == "__main__":
    success = run_all_tournament_tests()
    sys.exit(0 if success else 1)