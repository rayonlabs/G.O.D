import json
import os

import numpy as np
import safetensors.torch
from diffusers import StableDiffusionPipeline
from fiber.logging_utils import get_logger
from huggingface_hub import HfApi
from huggingface_hub import snapshot_download
from PIL import Image

from validator.core import constants as cst
from validator.core.models import Img2ImgPayload
from validator.evaluation.utils import adjust_image_size
from validator.evaluation.utils import base64_to_image
from validator.evaluation.utils import download_from_huggingface
from validator.evaluation.utils import image_to_base64
from validator.evaluation.utils import list_supported_images
from validator.evaluation.utils import read_prompt_file
from validator.utils import comfy_api_gate as api_gate


logger = get_logger(__name__)
hf_api = HfApi()


def load_comfy_workflows():
    with open(cst.LORA_WORKFLOW_PATH, "r") as file:
        lora_template = json.load(file)

    with open(cst.LORA_WORKFLOW_PATH_DIFFUSERS, "r") as file:
        lora_template_diffusers = json.load(file)

    return lora_template, lora_template_diffusers


def contains_image_files(directory: str) -> str:
    try:
        return any(file.lower().endswith(cst.SUPPORTED_IMAGE_FILE_EXTENSIONS) for file in os.listdir(directory))
    except FileNotFoundError:
        return False


def validate_dataset_path(dataset_path: str) -> str:
    if os.path.isdir(dataset_path):
        if contains_image_files(dataset_path):
            return dataset_path
        subdirectories = [
            os.path.join(dataset_path, d) for d in os.listdir(dataset_path) if os.path.isdir(os.path.join(dataset_path, d))
        ]
        for subdirectory in subdirectories:
            if contains_image_files(subdirectory):
                return subdirectory
    return dataset_path


def find_latest_lora_submission_name(repo_id: str) -> str:
    repo_files = hf_api.list_repo_files(repo_id)
    model_files = [file for file in repo_files if file.startswith(cst.DIFFUSION_HF_DEFAULT_FOLDER)]

    for file in model_files:
        if file.endswith(cst.DIFFUSION_HF_DEFAULT_CKPT_NAME):
            return file

    epoch_files = []
    for file in model_files:
        if "last-" in file and file.endswith(".safetensors"):
            try:
                epoch = int(file.split("last-")[1].split(".")[0])
                epoch_files.append((epoch, file))
            except ValueError:
                continue

    if epoch_files:
        epoch_files.sort(reverse=True, key=lambda x: x[0])
        return epoch_files[0][1]

    return None


def is_safetensors_available(repo_id: str) -> tuple[bool, str | None]:
    files_metadata = hf_api.list_repo_tree(repo_id=repo_id, repo_type="model")
    for file in files_metadata:
        if hasattr(file, "size") and file.size is not None:
            if file.path.endswith(".safetensors") and file.size > 6 * 1024 * 1024 * 1024:
                return True, file.path

    return False, None


def download_base_model(repo_id: str, safetensors_filename: str | None = None) -> str:
    if safetensors_filename:
        model_path = download_from_huggingface(repo_id, safetensors_filename, cst.CHECKPOINTS_SAVE_PATH)
    else:
        model_name = repo_id.split("/")[-1]
        save_dir = f"{cst.DIFFUSERS_PATH}/{model_name}"
        model_path = snapshot_download(repo_id=repo_id, local_dir=save_dir, repo_type="model")
        return model_name, model_path

    return safetensors_filename, model_path


def download_lora(repo_id: str) -> str:
    lora_save_name = repo_id.split("/")[-1]
    if not os.path.exists(f"{cst.LORAS_SAVE_PATH}/{lora_save_name}.safetensors"):
        lora_filename = find_latest_lora_submission_name(repo_id)
        local_path = download_from_huggingface(repo_id, lora_filename, cst.LORAS_SAVE_PATH)
        unique_path = f"{cst.LORAS_SAVE_PATH}/{lora_save_name}.safetensors"
        os.rename(local_path, unique_path)
        logger.info(f"Downloaded {unique_path}")
        return unique_path
    else:
        return f"{cst.LORAS_SAVE_PATH}/{lora_save_name}.safetensors"


def calculate_l2_loss(test_image: Image.Image, generated_image: Image.Image) -> float:
    test_image = np.array(test_image.convert("RGB")) / 255.0
    generated_image = np.array(generated_image.convert("RGB")) / 255.0
    if test_image.shape != generated_image.shape:
        raise ValueError("Images must have the same dimensions to calculate L2 loss.")
    l2_loss = np.mean((test_image - generated_image) ** 2)
    return l2_loss


def edit_workflow(payload: dict, edit_elements: Img2ImgPayload, text_guided: bool, is_safetensors: bool = True) -> dict:
    if is_safetensors:
        payload["Checkpoint_loader"]["inputs"]["ckpt_name"] = edit_elements.ckpt_name
    else:
        payload["Checkpoint_loader"]["inputs"]["model_path"] = edit_elements.ckpt_name
    payload["Sampler"]["inputs"]["steps"] = edit_elements.steps
    payload["Sampler"]["inputs"]["cfg"] = edit_elements.cfg
    payload["Sampler"]["inputs"]["denoise"] = edit_elements.denoise
    payload["Image_loader"]["inputs"]["image"] = edit_elements.base_image
    payload["Lora_loader"]["inputs"]["lora_name"] = edit_elements.lora_name
    if text_guided:
        payload["Prompt"]["inputs"]["text"] = edit_elements.prompt
    else:
        payload["Prompt"]["inputs"]["text"] = ""

    return payload


def inference(image_base64: str, params: Img2ImgPayload, use_prompt: bool = False, prompt: str = None) -> tuple[float, float]:
    if use_prompt and prompt:
        params.prompt = prompt

    params.base_image = image_base64

    lora_payload = edit_workflow(
        payload=params.comfy_template, edit_elements=params, text_guided=use_prompt, is_safetensors=params.is_safetensors
    )
    lora_gen = api_gate.generate(lora_payload)[0]
    lora_gen_loss = calculate_l2_loss(base64_to_image(image_base64), lora_gen)
    logger.info(f"Loss: {lora_gen_loss}")

    return lora_gen_loss


def eval_loop(dataset_path: str, params: Img2ImgPayload) -> dict[str, list[float]]:
    lora_losses_text_guided = []
    lora_losses_no_text = []

    test_images_list = list_supported_images(dataset_path, cst.SUPPORTED_IMAGE_FILE_EXTENSIONS)

    for file_name in test_images_list:
        logger.info(f"Calculating losses for {file_name}")

        base_name = os.path.splitext(file_name)[0]
        png_path = os.path.join(dataset_path, file_name)
        txt_path = os.path.join(dataset_path, f"{base_name}.txt")
        test_image = Image.open(png_path)
        test_image = adjust_image_size(test_image)
        image_base64 = image_to_base64(test_image)
        prompt = read_prompt_file(txt_path)

        params.prompt = prompt

        lora_losses_text_guided.append(inference(image_base64, params, use_prompt=True))
        lora_losses_no_text.append(inference(image_base64, params, use_prompt=False))

    return {"text_guided_losses": lora_losses_text_guided, "no_text_losses": lora_losses_no_text}


def _count_model_parameters(model_path: str, is_safetensors: bool) -> int:
    try:
        if is_safetensors:
            state_dict = safetensors.torch.load_file(model_path)
            return sum(p.numel() for p in state_dict.values()) or 0
        else:
            pipe = StableDiffusionPipeline.from_pretrained(model_path)
            total_params = 0
            for attr in pipe.__dict__.values():
                if hasattr(attr, "parameters"):
                    total_params += sum(p.numel() for p in attr.parameters())
            return total_params
    except Exception as e:
        logger.error(f"Failed to count model parameters: {e}")
        return 0


def main():
    test_dataset_path = os.environ.get("DATASET")
    base_model_repo = os.environ.get("ORIGINAL_MODEL_REPO")
    trained_lora_model_repos = os.environ.get("MODELS", "")
    if not all([test_dataset_path, base_model_repo, trained_lora_model_repos]):
        logger.error("Missing required environment variables.")
        exit(1)

    is_safetensors, safetensors_filename = is_safetensors_available(base_model_repo)
    # Base model download
    logger.info("Downloading base model")
    model_name_or_path, model_path = download_base_model(base_model_repo, safetensors_filename=safetensors_filename)
    logger.info("Base model downloaded")

    lora_repos = [m.strip() for m in trained_lora_model_repos.split(",") if m.strip()]

    test_dataset_path = validate_dataset_path(test_dataset_path)

    lora_comfy_template, diffusers_comfy_template = load_comfy_workflows()
    api_gate.connect()

    results = {"model_params_count": _count_model_parameters(model_path, is_safetensors)}

    for repo_id in lora_repos:
        try:
            lora_local_path = download_lora(repo_id)
            img2img_payload = Img2ImgPayload(
                ckpt_name=model_name_or_path,
                lora_name=os.path.basename(lora_local_path),
                steps=cst.DEFAULT_STEPS,
                cfg=cst.DEFAULT_CFG,
                denoise=cst.DEFAULT_DENOISE,
                comfy_template=lora_comfy_template if is_safetensors else diffusers_comfy_template,
                is_safetensors=is_safetensors,
            )

            loss_data = eval_loop(test_dataset_path, img2img_payload)
            results[repo_id] = {"eval_loss": loss_data}

            if os.path.exists(lora_local_path):
                os.remove(lora_local_path)
        except Exception as e:
            logger.error(f"Error evaluating repo {repo_id}: {str(e)}")
            results[repo_id] = str(e)

    output_file = "/aplp/evaluation_results.json"
    output_dir = os.path.dirname(output_file)

    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    with open(output_file, "w") as f:
        json.dump(results, f)

    logger.info(f"Evaluation results saved to {output_file}")

    logger.info(json.dumps(results))


if __name__ == "__main__":
    main()
