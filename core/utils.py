import json

import pandas as pd
from loguru import logger

from core.models.utility_models import DatasetType
from core.models.utility_models import FileFormat


async def validate_dataset(
    dataset_path: str, dataset_type: DatasetType, file_format: FileFormat
) -> bool:
    try:
        if file_format == FileFormat.CSV:
            df = pd.read_csv(dataset_path)
        elif file_format == FileFormat.JSON:
            logger.info("now is json")
            with open(dataset_path, "r") as f:
                data = json.load(f)
            df = pd.DataFrame(data)
        else:
            raise ValueError(f"Unsupported file format: {file_format}")

        if dataset_type == DatasetType.INSTRUCT:
            required_columns = ["instruction", "input", "output"]
        else:
            required_columns = ["input"]

        return all(col in df.columns for col in required_columns)
    except Exception as e:
        raise ValueError(f"Error validating dataset: {str(e)}")
