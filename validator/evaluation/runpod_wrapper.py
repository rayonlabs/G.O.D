import runpod
from validator.evaluation.eval_diffusion import evaluate

def handler(job):
    eval_details = job["input"]
    response = evaluate(
        test_dataset_zip=eval_details["test_dataset_zip"],
        base_model_repo=eval_details["base_model_repo"],
        trained_lora_model_repos=eval_details["trained_lora_model_repos"],
        model_type=eval_details["model_type"],
    )
    return response

runpod.serverless.start({"handler": handler})