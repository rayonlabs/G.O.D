import yaml
from const import CONFIG_TEMPLATE_PATH, HUGGINGFACE_TOKEN
from schemas import DatasetType, FileFormat

import logging

logger = logging.getLogger(__name__)

def load_and_modify_config(job_id: str, dataset: str, model: str, dataset_type: DatasetType, file_format: FileFormat) -> dict:
    with open(CONFIG_TEMPLATE_PATH, 'r') as file:
        config = yaml.safe_load(file)

    config['datasets'][0]['path'] = dataset
    config['base_model'] = model
    config['base_model_config'] = model
    config['datasets'][0]['type'] = dataset_type.value  
    config['datasets'][0]['ds_type'] = file_format.value if file_format != FileFormat.HF else None
    return config

def save_config(config: dict, config_path: str):
    with open(config_path, 'w') as file:
        yaml.dump(config, file)

