import sys
import traceback

from validator.evaluation.eval_dpo import evaluate_dpo_repo
from validator.utils.logging import get_logger


logger = get_logger(__name__)

if __name__ == "__main__":
    try:
        # Check command line arguments
        if len(sys.argv) != 6:
            logger.error(f"Expected 5 arguments, got {len(sys.argv) - 1}")
            logger.error(
                "Usage: python -m validator.evaluation.eval_dpo_single \n"
                "       <repo> <dataset> <original_model> <dataset_type> <file_format>"
            )
            sys.exit(1)

        # Get command line arguments
        repo = sys.argv[1]
        dataset = sys.argv[2]
        original_model = sys.argv[3]
        dataset_type_str = sys.argv[4]
        file_format_str = sys.argv[5]

        logger.info(f"Starting evaluation for {repo}")
        logger.info(f"Arguments: dataset={dataset}, original_model={original_model}")
        logger.info(f"dataset_type={dataset_type_str}, file_format={file_format_str}")

        # Run evaluation
        evaluate_dpo_repo(repo, dataset, original_model, dataset_type_str, file_format_str)

        logger.info(f"Evaluation for {repo} completed")
    except Exception as e:
        logger.critical(f"Unhandled exception: {e}")
        logger.critical(traceback.format_exc())
        sys.exit(1)
