import os
import time
from dataclasses import dataclass

import httpx
import websocket
from dotenv import load_dotenv
from fiber.chain import chain_utils
from fiber.chain import interface
from redis.asyncio import Redis
from substrateinterface import Keypair
from substrateinterface import SubstrateInterface

from validator.db.database import PSQLDB
from validator.utils.logging import get_logger


logger = get_logger(__name__)
load_dotenv(".vali.env")


@dataclass
class Config:
    substrate: SubstrateInterface
    keypair: Keypair
    psql_db: PSQLDB
    redis_db: Redis
    subtensor_network: str | None
    subtensor_address: str | None
    netuid: int
    refresh_nodes: bool
    httpx_client: httpx.AsyncClient
    set_metagraph_weights_with_high_updated_to_not_dereg: bool
    s3_compatible_endpoint: str
    s3_compatible_access_key: str
    s3_compatible_secret_key: str
    s3_bucket_name: str
    s3_region: str
    testnet: bool = os.getenv("SUBTENSOR_NETWORK", "").lower() == "test"
    debug: bool = os.getenv("ENV", "prod").lower() != "prod"


_config = None


def load_config() -> Config:
    global _config
    if _config is None:
        subtensor_network = os.getenv("SUBTENSOR_NETWORK")
        subtensor_address = os.getenv("SUBTENSOR_ADDRESS") or None
        dev_env = os.getenv("ENV", "prod").lower() != "prod"
        wallet_name = os.getenv("WALLET_NAME", "default")
        hotkey_name = os.getenv("HOTKEY_NAME", "default")
        netuid = os.getenv("NETUID")
        if netuid is None:
            netuid = 201 if subtensor_network == "test" else 69420
            logger.warning(f"NETUID not set, using {netuid}")
        else:
            netuid = int(netuid)

        redis_host = "localhost"

        refresh_nodes: bool = os.getenv("REFRESH_NODES", "true").lower() == "true"
        if refresh_nodes:
            try:
                substrate = interface.get_substrate(subtensor_network=subtensor_network, subtensor_address=subtensor_address)
            except websocket._exceptions.WebSocketBadStatusException as e:
                logger.error(f"Failed to get substrate: {e}. Sleeping for 20 seconds and then trying again...")
                time.sleep(20)
                substrate = interface.get_substrate(subtensor_network=subtensor_network, subtensor_address=subtensor_address)
        else:
            # this is only used for testing
            substrate = None
        keypair = chain_utils.load_hotkey_keypair(wallet_name=wallet_name, hotkey_name=hotkey_name)
        logger.info(f"This is my own keypair {keypair}")

        httpx_limits = httpx.Limits(max_connections=500, max_keepalive_connections=100)
        httpx_client = httpx.AsyncClient(limits=httpx_limits)

        set_metagraph_weights_with_high_updated_to_not_dereg = bool(
            os.getenv("SET_METAGRAPH_WEIGHTS_WITH_HIGH_UPDATED_TO_NOT_DEREG", "false").lower() == "true"
        )

        s3_compatible_endpoint = os.getenv("S3_COMPATIBLE_ENDPOINT", "localhost:9000")
        s3_compatible_access_key = os.getenv("S3_COMPATIBLE_ACCESS_KEY", "minioadmin")
        s3_compatible_secret_key = os.getenv("S3_COMPATIBLE_SECRET_KEY", "minioadmin")
        s3_bucket_name = os.getenv("S3_BUCKET_NAME")
        s3_region = os.getenv("S3_REGION", "us-east-1")

        _config = Config(
            substrate=substrate,
            keypair=keypair,
            psql_db=PSQLDB(),
            redis_db=Redis(host=redis_host),
            subtensor_network=subtensor_network,
            subtensor_address=subtensor_address,
            netuid=netuid,
            refresh_nodes=refresh_nodes,
            httpx_client=httpx_client,
            debug=dev_env,
            set_metagraph_weights_with_high_updated_to_not_dereg=set_metagraph_weights_with_high_updated_to_not_dereg,
            s3_compatible_endpoint=s3_compatible_endpoint,
            s3_compatible_access_key=s3_compatible_access_key,
            s3_compatible_secret_key=s3_compatible_secret_key,
            s3_bucket_name=s3_bucket_name,
            s3_region=s3_region,
        )
    return _config
