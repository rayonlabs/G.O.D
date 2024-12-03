import asyncio
import json
import os
import time
from typing import Any
from typing import Optional

import httpx
import netaddr
import requests
import urllib3
from fiber import Keypair
from fiber.chain import chain_utils
from fiber.chain.models import Node
from fiber.logging_utils import get_logger
from fiber.validator import client
from tenacity import retry
from tenacity import retry_if_exception_type
from tenacity import stop_after_attempt
from tenacity import wait_exponential

from core.constants import NETUID
from validator.core.config import Config
from validator.core.constants import GRADIENTS_ENDPOINT
from validator.core.constants import NINETEEN_API_KEY
from validator.core.constants import PROMPT_GEN_ENDPOINT


logger = get_logger(__name__)

# Create a retry decorator with exponential backoff
retry_with_backoff = retry(
    stop=stop_after_attempt(5),
    wait=wait_exponential(multiplier=1, min=4, max=10),
    retry=retry_if_exception_type((httpx.HTTPStatusError, httpx.RequestError)),
    reraise=True,
)


def _get_headers_for_signed_https_request(keypair: Keypair):
    nonce = f"{time.time_ns()}"
    signature = chain_utils.sign_message(keypair, nonce)
    headers = {
        "validator-hotkey": keypair.ss58_address,
        "signature": signature,
        "nonce": nonce,
        "netuid": str(NETUID),
        "Content-Type": "application/json",
    }
    return headers


async def process_non_stream_fiber_get(endpoint: str, config: Config, node: Node) -> dict[str, Any] | None:
    server_address = client.construct_server_address(
        node=node,
        replace_with_docker_localhost=False,
        replace_with_localhost=False,
    )
    logger.info(
        f"Attempting to hit a GET {server_address} endpoint {endpoint}")
    try:
        response = await client.make_non_streamed_get(
            httpx_client=config.httpx_client,
            server_address=server_address,
            validator_ss58_address=config.keypair.ss58_address,
            endpoint=endpoint,
            timeout=10,
        )
    except Exception as e:
        logger.error(f"Failed to communicate with node {node.node_id}: {e}")
        return None

    if response.status_code != 200:
        logger.warning(f"Failed to communicate with node {node.node_id}")
        return None

    return response.json()


async def process_non_stream_fiber(
    endpoint: str, config: Config, node: Node, payload: dict[str, Any], timeout: int = 10
) -> dict[str, Any] | None:
    server_address = client.construct_server_address(
        node=node,
        replace_with_docker_localhost=False,
        replace_with_localhost=False,
    )
    try:
        response = await client.make_non_streamed_post(
            httpx_client=config.httpx_client,
            server_address=server_address,
            validator_ss58_address=config.keypair.ss58_address,
            miner_ss58_address=node.hotkey,
            keypair=config.keypair,
            endpoint=endpoint,
            payload=payload,
            timeout=timeout,
        )
    except Exception as e:
        logger.debug(f"Failed to communicate with node {node.node_id}: {e}")
        return None

    if response.status_code != 200:
        logger.debug(f"Failed to communicate with node {node.node_id}")
        return None

    return response.json()


@retry_with_backoff
async def post_to_nineteen_ai(payload: dict[str, Any], keypair: Keypair) -> str | None:
    if NINETEEN_API_KEY is None:
        headers = _get_headers_for_signed_https_request(keypair)
    else:
        headers = {
            "Authorization": f"Bearer {NINETEEN_API_KEY}",
            "Content-Type": "application/json",
        }

    async with httpx.AsyncClient(timeout=120) as client:
        response = await client.post(url=PROMPT_GEN_ENDPOINT, json=payload, headers=headers)
        if response.status_code != 200:
            logger.error(f"Error in nineteen ai response: {response.content}")
            response.raise_for_status()
        response_json = response.json()
        try:
            return response_json["choices"][0]["message"]["content"]
        except (KeyError, IndexError) as e:
            logger.error(f"Error in nineteen ai response: {response_json}")
            logger.exception(e)
            return None


# If this it to talk to the miner, its already in fiber
# We can change to that once we add bittensor stuff (i know that's why its like this ATM)
@retry_with_backoff
async def process_non_stream_get(base_url: str, token: Optional[str]) -> dict[str, Any] | list[dict[str, Any]]:
    headers = {
        "Accept": "application/json",
        "Authorization": f"Bearer {token}",
    }

    async with httpx.AsyncClient(timeout=120) as client:
        response = await client.get(base_url, headers=headers)
        response.raise_for_status()
        return response.json()


def _ip_to_int(str_val: str) -> int:
    return int(netaddr.IPAddress(str_val))


#  Shamelessly stole this from bittensor code
def get_external_ip() -> str:
    """Checks CURL/URLLIB/IPIFY/AWS for your external ip.
    Returns:
        external_ip  (:obj:`str` `required`):
            Your routers external facing ip as a string.

    Raises:
        ExternalIPNotFound (Exception):
            Raised if all external ip attempts fail.
    """
    # --- Try AWS
    try:
        external_ip = requests.get(
            "https://checkip.amazonaws.com").text.strip()
        assert isinstance(_ip_to_int(external_ip), int)
        return str(external_ip)
    except Exception:
        pass

    # --- Try ipconfig.
    try:
        process = os.popen("curl -s ifconfig.me")
        external_ip = process.readline()
        process.close()
        assert isinstance(_ip_to_int(external_ip), int)
        return str(external_ip)
    except Exception:
        pass

    # --- Try ipinfo.
    try:
        process = os.popen("curl -s https://ipinfo.io")
        external_ip = json.loads(process.read())["ip"]
        process.close()
        assert isinstance(_ip_to_int(external_ip), int)
        return str(external_ip)
    except Exception:
        pass

    # --- Try myip.dnsomatic
    try:
        process = os.popen("curl -s myip.dnsomatic.com")
        external_ip = process.readline()
        process.close()
        assert isinstance(_ip_to_int(external_ip), int)
        return str(external_ip)
    except Exception:
        pass

    # --- Try urllib ipv6
    try:
        external_ip = urllib3.request.urlopen(
            "https://ident.me").read().decode("utf8")
        assert isinstance(_ip_to_int(external_ip), int)
        return str(external_ip)
    except Exception:
        pass

    # --- Try Wikipedia
    try:
        external_ip = requests.get(
            "https://www.wikipedia.org").headers["X-Client-IP"]
        assert isinstance(_ip_to_int(external_ip), int)
        return str(external_ip)
    except Exception:
        pass

    raise Exception("Failed to get external IP!! :(")


async def sign_up_to_gradients(keypair: Keypair):
    headers = _get_headers_for_signed_https_request(keypair)

    frontend_api_key = os.getenv("FRONTEND_API_KEY")
    if frontend_api_key is None:
        raise ValueError("FRONTEND_API_KEY is not set!!")

    validator_port = os.getenv("VALIDATOR_PORT")
    if validator_port is None:
        raise ValueError("VALIDATOR_PORT is not set!!")

    request_body = {
        "hotkey": keypair.ss58_address,
        "netuid": str(NETUID),
        "ip_address": get_external_ip().rstrip(":") + ":" + validator_port,
        "frontend_api_key": frontend_api_key,
    }
    logger.info(f"Signing up to Gradients API with {request_body}")
    async with httpx.AsyncClient(timeout=120) as client:
        response = await client.post(url=GRADIENTS_ENDPOINT, headers=headers, json=request_body)
        if response.status_code == 409:
            logger.info("Already signed up to Gradients API!")
            return response.json()
        if response.status_code != 200:
            raise Exception(
                f"Failed to sign up to Gradients API with status code {response.status_code} and response {response.json()}!"
            )

        logger.info(
            f"Signed up to Gradients API successfully with response {response.json()}!")
        return response.json()


async def sign_up_cron_job(keypair: Keypair):
    # In case initial signup fails, we try again every 3 hours
    while True:
        await sign_up_to_gradients(keypair)
        await asyncio.sleep(60 * 60 * 24)  # 3 hours
