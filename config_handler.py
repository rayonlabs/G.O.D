import yaml
from const import WANDB_PROJECT, WANDB_ENTITY, CONFIG_TEMPLATE_PATH
from schemas import DatasetType, FileFormat

def load_and_modify_config(job_id: str, dataset: str, model: str, dataset_type: DatasetType, file_format: FileFormat) -> dict:
    with open(CONFIG_TEMPLATE_PATH, 'r') as file:
        config = yaml.safe_load(file)

    config['datasets'][0]['path'] = dataset
    config['base_model'] = model
    config['base_model_config'] = model
    config['dataset_type'] = dataset_type.value
    config['file_format'] = file_format.value
    config['wandb_project'] = WANDB_PROJECT
    config['wandb_entity'] = WANDB_ENTITY

    return config

def save_config(config: dict, config_path: str):
    with open(config_path, 'w') as file:
        yaml.dump(config, file)

