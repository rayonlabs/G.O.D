import argparse
import asyncio
import os
from logging import getLogger
from typing import Optional

import asyncpg
from dotenv import load_dotenv
from fiber.chain.models import Node


logger = getLogger(__name__)


async def inject_dev_miner(
    hotkey: str,
    coldkey: str,
    node_id: int,
    ip: str = "localhost",
    port: int = 7999,
    netuid: int = 241,
    db_name: Optional[str] = None,
    db_user: Optional[str] = None,
    db_password: Optional[str] = None,
    db_host: Optional[str] = None,
) -> None:
    """Inject a development miner into the validator's database."""
    if not all([db_name, db_user, db_password, db_host]):
        load_dotenv(".vali.env")
        db_name = os.getenv("POSTGRES_DB")
        db_user = os.getenv("POSTGRES_USER")
        db_password = os.getenv("POSTGRES_PASSWORD")
        db_host = os.getenv("POSTGRES_HOST")
        logger.info(f"Using database {db_name} on {db_host} with user {db_user} and password {db_password}")

    conn = await asyncpg.connect(
        database=db_name,
        user=db_user,
        password=db_password,
        host=db_host,
    )

    node = Node(
        hotkey=hotkey,
        ip=ip,
        port=port,
        netuid=netuid,
        node_id=node_id,
        incentive=1.0,
        stake=1000.0,
        alpha_stake=500.0,
        tao_stake=500.0,
        coldkey=coldkey,
        trust=0.0,
        vtrust=0.0,
        last_updated=0,
        ip_type=4,
    )

    await conn.execute(
        """
        INSERT INTO nodes (
            hotkey, ip, port, netuid, node_id, incentive, stake, 
            alpha_stake, tao_stake, coldkey, ip_type, trust, vtrust, last_updated
        )
        VALUES ($1, $2, $3, $4, $5, $6, $7, $8, $9, $10, $11, $12, $13, $14)
        ON CONFLICT (hotkey, netuid) 
        DO UPDATE SET 
            ip = $2,
            port = $3,
            node_id = $5,
            incentive = $6,
            stake = $7,
            alpha_stake = $8,
            tao_stake = $9,
            coldkey = $10,
            ip_type = $11,
            trust = $12,
            vtrust = $13,
            last_updated = $14
        """,
        node.hotkey,
        node.ip,
        node.port,
        node.netuid,
        node.node_id,
        node.incentive,
        node.stake,
        node.alpha_stake,
        node.tao_stake,
        node.coldkey,
        node.ip_type,
        node.trust,
        node.vtrust,
        node.last_updated,
    )

    await conn.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Inject a development miner into the validator database.")
    parser.add_argument("--hotkey", help="The hotkey for the miner")
    parser.add_argument("--coldkey", help="The coldkey for the miner")
    parser.add_argument("--node_id", type=int, help="Node ID for the miner")
    parser.add_argument("--ip", default="localhost", help="IP address (default: localhost)")
    parser.add_argument("--port", type=int, default=7999, help="Port number (default: 7999)")
    parser.add_argument("--netuid", type=int, default=241, help="Network UID (default: 241)")

    args = parser.parse_args()

    asyncio.run(inject_dev_miner(hotkey=args.hotkey, coldkey=args.coldkey, ip=args.ip, port=args.port, netuid=args.netuid))
