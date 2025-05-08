import json
import os
import sys

import torch
from huggingface_hub import HfApi
from huggingface_hub.utils import EntryNotFoundError
from huggingface_hub.utils import HfHubHTTPError
from transformers import AutoModelForCausalLM
from transformers import AutoTokenizer

from validator.core import constants as cst
from validator.utils.logging import get_logger


logger = get_logger(__name__)


def count_parameters(model: torch.nn.Module) -> int:
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def run_check():
    model_id = os.environ.get("MODEL_ID")

    results = {
        "model_id": model_id,
        "status": "Failure",
        "parameter_count": None,
        "error_message": None,
    }

    if not model_id:
        results["error_message"] = "MODEL_ID environment variable not set."
        logger.error(results["error_message"])
        with open(cst.CHECK_RESULTS_PATH, "w") as f:
            json.dump(results, f)
        sys.exit(1)

    logger.info(f"--- Checking model: {model_id} ---")
    model = None
    tokenizer = None

    try:
        param_count_from_api = False
        try:
            logger.info("Attempting to fetch parameter count from Hub API...")
            api = HfApi()
            model_info = api.model_info(model_id)
            if (
                hasattr(model_info, "safetensors")
                and model_info.safetensors
                and hasattr(model_info.safetensors, "total")
                and model_info.safetensors.total is not None
            ):
                param_count_api = model_info.safetensors.total
                results["parameter_count"] = param_count_api
                param_count_from_api = True
                logger.info(f"Parameter count from API (safetensors.total): {param_count_api:,}")
            else:
                logger.warning("Could not find 'safetensors.total' in model metadata. Will attempt fallback count after loading.")
        except (EntryNotFoundError, HfHubHTTPError, Exception) as e:
            logger.warning(
                f"Failed to get parameter count from API: {type(e).__name__} - {str(e)}. Will attempt fallback count after loading."
            )

        logger.info("Loading tokenizer...")
        tokenizer = AutoTokenizer.from_pretrained(model_id, trust_remote_code=False)
        if tokenizer.pad_token_id is None and tokenizer.eos_token_id is not None:
            logger.info("Setting pad_token to eos_token as pad_token is None.")
            tokenizer.pad_token = tokenizer.eos_token
        logger.info("Tokenizer loaded successfully.")

        logger.info("Loading model...")
        model = AutoModelForCausalLM.from_pretrained(
            model_id,
            torch_dtype=torch.bfloat16,
            device_map="auto",
            trust_remote_code=False,
        )
        logger.info("Model loaded successfully.")
        model.eval()

        if not param_count_from_api:
            logger.info("Attempting fallback parameter count using loaded model...")
            try:
                counted_params = count_parameters(model)
                results["parameter_count"] = counted_params
                logger.info(f"Parameter count from loaded model: {counted_params:,}")
            except Exception as e:
                logger.warning(f"Failed to count parameters from loaded model: {e}")

        logger.info("Performing inference check...")
        test_input = "Hello, how are you?"
        inputs = tokenizer(test_input, return_tensors="pt").to(model.device)

        gen_kwargs = {"max_new_tokens": 5}
        pad_token_id_to_use = None
        if tokenizer.pad_token_id is not None:
            pad_token_id_to_use = tokenizer.pad_token_id
        elif tokenizer.eos_token_id is not None:
            logger.warning("pad_token_id not set, using eos_token_id for generation padding.")
            pad_token_id_to_use = tokenizer.eos_token_id

        if pad_token_id_to_use is not None:
            gen_kwargs["pad_token_id"] = pad_token_id_to_use
        else:
            logger.warning("Neither pad_token_id nor eos_token_id is set for the tokenizer.")

        with torch.no_grad():
            outputs = model.generate(**inputs, **gen_kwargs)

        result_text = tokenizer.decode(outputs[0], skip_special_tokens=True)
        logger.info(f"Inference successful. Input: '{test_input}', Output: '{result_text}'")

        results["status"] = "Success"
        logger.info(f"Model check successful for {model_id}.")
        with open(cst.CHECK_RESULTS_PATH, "w") as f:
            json.dump(results, f)
        sys.exit(0)

    except Exception as e:
        error_msg = f"Error during model check for {model_id}: {type(e).__name__} - {str(e)}"
        results["error_message"] = error_msg
        logger.error(error_msg, exc_info=True)
        try:
            with open(cst.CHECK_RESULTS_PATH, "w") as f:
                json.dump(results, f)
        except Exception as write_err:
            logger.error(f"Additionally failed to write error results to {cst.CHECK_RESULTS_PATH}: {write_err}")
        sys.exit(1)
    finally:
        del model
        del tokenizer
        if torch.cuda.is_available():
            torch.cuda.empty_cache()


if __name__ == "__main__":
    run_check()
