import sys

from validator.evaluation.eval_dpo import evaluate_dpo_repo
from validator.utils.logging import get_logger


logger = get_logger(__name__)

if __name__ == "__main__":
    if len(sys.argv) != 6:
        logger.error(f"Expected 5 arguments, got {len(sys.argv) - 1}")
        logger.error(
            "Usage: python -m validator.evaluation.single_eval_dpo \n"
            "       <repo> <dataset> <original_model> <dataset_type> <file_format>"
        )
        sys.exit(1)

    evaluate_dpo_repo(
        repo=sys.argv[1],
        dataset=sys.argv[2],
        original_model=sys.argv[3],
        dataset_type_str=sys.argv[4],
        file_format_str=sys.argv[5],
    )
