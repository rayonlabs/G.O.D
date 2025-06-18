from datetime import datetime
from unittest.mock import Mock
from uuid import uuid4
from pydantic import BaseModel
from fiber.chain.models import Node
from validator.core.config import Config


def create_test_node(hotkey: str) -> Node:
    return Node(
        hotkey=hotkey,
        coldkey="test_coldkey",
        node_id=0,
        incentive=0.0,
        netuid=181,
        alpha_stake=0.0,
        tao_stake=0.0,
        stake=0.0,
        trust=0.0,
        vtrust=0.0,
        last_updated=0.0,
        ip="0.0.0.0",
        ip_type=4,
        port=8080,
        protocol=4
    )


def create_test_config() -> Config:
    """Create a minimal mock config for testing"""
    mock_substrate = Mock()
    mock_keypair = Mock()
    mock_psql_db = MockPSQLDB()
    mock_redis_db = Mock()
    mock_httpx_client = Mock()
    
    return Config(
        substrate=mock_substrate,
        keypair=mock_keypair,
        psql_db=mock_psql_db,
        redis_db=mock_redis_db,
        subtensor_network="test",
        subtensor_address="test",
        netuid=181,
        refresh_nodes=False,
        httpx_client=mock_httpx_client,
        set_metagraph_weights_with_high_updated_to_not_dereg=False
    )


class MockPSQLDB:
    def __init__(self, config=None):
        pass
    
    async def connection(self):
        return MockConnection()


class MockConnection:
    def __init__(self):
        pass
    
    async def __aenter__(self):
        return self
    
    async def __aexit__(self, exc_type, exc_val, exc_tb):
        pass
    
    async def fetchrow(self, query, *args):
        return {"tournament_id": "mock_tournament", "status": "pending"}
    
    async def fetch(self, query, *args):
        return []
    
    async def execute(self, query, *args):
        pass
    
    def transaction(self):
        return MockTransaction()


class MockTransaction:
    async def __aenter__(self):
        return self
    
    async def __aexit__(self, exc_type, exc_val, exc_tb):
        pass