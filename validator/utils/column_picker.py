import json
import re
from logging import getLogger

from fastapi import HTTPException

from validator.core.config import Config
from validator.core.constants import COLUMN_PICKER_NUM_PREVIEW_ROWS
from validator.core.constants import SYNTH_MODEL
from validator.core.constants import SYNTH_MODEL_TEMPERATURE
from validator.core.models import ColumnPickerResponse
from validator.core.models import SuitableDataset
from validator.utils.call_endpoint import post_to_nineteen_ai
from validator.utils.prompts import load_prompts


logger = getLogger(__name__)


def create_prompt(dataset: SuitableDataset) -> str:
    if not dataset.preview:
        raise HTTPException(
            status_code=422, detail="Dataset exists but contains no preview data, which is required for column picking"
        )

    prompts = load_prompts()

    prompt_template = prompts.auto_column_pick
    if not prompt_template:
        raise HTTPException(status_code=503, detail="Column picker service is not properly configured: PROMPT template missing")

    preview_rows = remove_invalid_columns(dataset.preview, dataset.sparse_columns + dataset.non_sparse_columns)
    preview_rows = preview_rows[: min(COLUMN_PICKER_NUM_PREVIEW_ROWS, len(preview_rows))]
    preview_str = json.dumps(preview_rows, indent=2)

    return prompt_template.format(
        dataset_name=dataset.dataset_id,
        sparse_columns=dataset.sparse_columns,
        dense_columns=dataset.non_sparse_columns,
        preview_data=preview_str,
    )


def remove_invalid_columns(preview: list[dict], valid_columns: list[str]) -> list[dict]:
    return [{k: v for k, v in row.items() if k in valid_columns} for row in preview]


async def pick_columns_locally(
    config: Config,
    dataset: SuitableDataset,
) -> ColumnPickerResponse:
    prompt = create_prompt(dataset)
    payload = {
        "model": SYNTH_MODEL,
        "temperature": SYNTH_MODEL_TEMPERATURE,
        "messages": [{"role": "user", "content": prompt}],
    }

    try:
        response = await post_to_nineteen_ai(payload, config.keypair)
        json_match = re.search(r"\{[\s\S]*\}", response)
        if json_match:
            return ColumnPickerResponse.model_validate_json(json_match.group())
        logger.debug(f"Did not get JSON response from LLM: {response}")
        raise HTTPException(status_code=422, detail="LLM failed to generate a valid response for column suggestions.")
    except Exception as e:
        logger.error(f"LLM auto column picker error: {e}")
        raise HTTPException(status_code=422, detail="LLM failed to generate a valid response for column suggestions.") from e
