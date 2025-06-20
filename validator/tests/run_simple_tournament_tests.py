#!/usr/bin/env python3

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from validator.tests.test_tournament_organiser import test_all_organiser_functionality
from validator.utils.logging import get_logger

logger = get_logger(__name__)


def run_simple_tournament_tests():
    logger.info("Starting simple tournament test suite...")
    
    logger.info("Running tournament organiser tests...")
    organiser_result = test_all_organiser_functionality()
    
    logger.info(f"\n{'='*50}")
    logger.info("TOURNAMENT TEST RESULTS:")
    logger.info(f"{'='*50}")
    
    status = "PASSED" if organiser_result else "FAILED"
    logger.info(f"Organiser tests: {status}")
    
    if organiser_result:
        logger.info("🎉 Tournament organiser tests passed!")
        return True
    else:
        logger.error("❌ Tournament organiser tests failed!")
        return False


if __name__ == "__main__":
    success = run_simple_tournament_tests()
    sys.exit(0 if success else 1)