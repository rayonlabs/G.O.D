import asyncio
from validator.evaluation.docker_evaluation import run_evaluation_docker_image
from core.models.utility_models import ImageModelType

def _calculate_weighted_loss_for_image_eval(eval_result) -> float:
    text_guided_avg = (
        sum(eval_result["eval_loss"]["text_guided_losses"]) / len(eval_result["eval_loss"]["text_guided_losses"])
        if eval_result["eval_loss"]["text_guided_losses"]
        else 0
    )

    no_text_avg = (
        sum(eval_result["eval_loss"]["no_text_losses"]) / len(eval_result["eval_loss"]["no_text_losses"])
        if eval_result["eval_loss"]["no_text_losses"]
        else 0
    )

    weighted_loss = (
        0.7 * text_guided_avg + (1 - 0.7) * no_text_avg
    )
    return weighted_loss

def main():
    loop = asyncio.get_event_loop()
    results = loop.run_until_complete(run_evaluation_docker_image(
        test_split_url="https://gradients.s3.eu-north-1.amazonaws.com/65436375d29320fa_test_data.zip?X-Amz-Algorithm=AWS4-HMAC-SHA256&X-Amz-Credential=AKIAVVZOOA7SA4UOFLPI%2F20250426%2Feu-north-1%2Fs3%2Faws4_request&X-Amz-Date=20250426T014527Z&X-Amz-Expires=604800&X-Amz-SignedHeaders=host&X-Amz-Signature=5d89c92c20782107b90c9e109ead4a32f44492ef49e8c69eb7fcc72b3e55b036",
        models=["tungdqzenai/b127b9b8-f094-4f5f-ab99-14b41d95946d"],
        original_model_repo="dataautogpt3/CALAMITY",
        model_type=ImageModelType.SDXL,
        gpu_ids = [0]
    ))
    print(results)

if __name__ == "__main__":
    main()
    # results = {"eval_loss": {"text_guided_losses": [0.08824672371427876, 0.0959852661500446], "no_text_losses": [0.07125932217072053, 0.04801038510582775]}}
    # print(_calculate_weighted_loss_for_image_eval(results))