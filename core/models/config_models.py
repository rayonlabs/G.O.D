from dataclasses import dataclass


@dataclass
class BaseConfig:
    wallet_name: str
    hotkey_name: str
    subtensor_network: str
    netuid: int
    env: str
    subtensor_address: str | None


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
    postgres_user: str | None = None
    postgres_password: str | None = None
    postgres_db: str | None = None
    postgres_host: str | None = None
    postgres_port: str | None = None
    s3_compatible_endpoint: str
    s3_compatible_access_key: str
    s3_compatible_secret_key: str
    s3_bucket_name: str
    frontend_api_key: str
    validator_port: str
    set_metagraph_weights: bool
    validator_port: str
    gpu_ids: str
    trainer_ips: str
    trainer_gpu_ids: str
    gpu_server: str | None = None
    localhost: bool = False
    env_file: str = ".vali.env"
    hf_datasets_trust_remote_code = True
    s3_region: str = "us-east-1"
    refresh_nodes: bool = True
    database_url: str | None = None
    postgres_profile: str = "default"

    def __post_init__(self):
        # Validate that trainer IPs and trainer GPU IDs have matching lengths
        if self.trainer_gpu_ids and self.trainer_ips:
            gpu_groups = [group.strip() for group in self.trainer_gpu_ids.split(";") if group.strip()]
            ips = [ip.strip() for ip in self.trainer_ips.split(",") if ip.strip()]

            if len(gpu_groups) != len(ips):
                raise ValueError(
                    f"Number of trainer GPU groups ({len(gpu_groups)}) must match "
                    f"number of trainer IPs ({len(ips)}). "
                    f"GPU groups: {gpu_groups}, IPs: {ips}"
                )


@dataclass
class AuditorConfig(BaseConfig): ...
