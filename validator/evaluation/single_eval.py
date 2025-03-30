import sys

from validator.evaluation.eval import evaluate_repo


if __name__ == "__main__":
    if len(sys.argv) != 6:
        print("Usage: python single_eval.py <repo> <dataset> <original_model> <dataset_type_str> <file_format_str>")
        sys.exit(1)

    evaluate_repo(
        repo=sys.argv[1],
        dataset=sys.argv[2],
        original_model=sys.argv[3],
        dataset_type_str=sys.argv[4],
        file_format_str=sys.argv[5],
    )
