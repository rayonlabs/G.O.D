from dataclasses import dataclass
from typing import Optional


@dataclass
class BaseConfig:
    wallet_name: str
    hotkey_name: str
    subtensor_network: str
    subtensor_address: Optional[str]
    netuid: int
    env: str


@dataclass
class MinerConfig(BaseConfig):
    wandb_token: str
    huggingface_username: str
    huggingface_token: str
    min_stake_threshold: str
    refresh_nodes: bool
    is_validator: bool = False


@dataclass
class ValidatorConfig(BaseConfig):
    postgres_user: str
    postgres_password: str
    postgres_db: str
    postgres_host: str
    postgres_port: str
    s3_compatible_endpoint: str
    s3_compatible_access_key: str
    s3_compatible_secret_key: str
    s3_bucket_name: str
    frontend_api_key: str
    validator_port: str
    set_metagraph_weights: bool
    validator_port: str
    gpu_ids: str
    gpu_server: Optional[str] = None
    localhost: bool = False
    env_file: str = ".vali.env"
    hf_datasets_trust_remote_code = True
    s3_region: str = "us-east-1"
    refresh_nodes: bool = True


@dataclass
class AuditorConfig(BaseConfig): ...
