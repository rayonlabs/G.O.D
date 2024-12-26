import datetime
from logging import getLogger
from typing import List
from typing import Optional

from asyncpg.connection import Connection
from fiber import SubstrateInterface
from fiber.chain.models import Node

from core.constants import NETUID
from validator.db import constants as dcst
from validator.db.database import PSQLDB
from validator.utils.query_substrate import query_substrate


logger = getLogger(__name__)


async def get_all_nodes(psql_db: PSQLDB) -> List[Node]:
    """Get all nodes for the current NETUID"""
    logger.info("Attempting to get all nodes")
    async with await psql_db.connection() as connection:
        connection: Connection
        query = f"""
            SELECT * FROM {dcst.NODES_TABLE}
            WHERE {dcst.NETUID} = $1
        """
        rows = await connection.fetch(query, NETUID)
        nodes = [Node(**dict(row)) for row in rows]
        logger.info(f"Here is the list of nodes {nodes}")
        return nodes


async def insert_nodes(connection: Connection, nodes: list[Node]) -> None:
    logger.info(f"Inserting {len(nodes)} nodes into {dcst.NODES_TABLE}...")
    await connection.executemany(
        f"""
        INSERT INTO {dcst.NODES_TABLE} (
            {dcst.HOTKEY},
            {dcst.COLDKEY},
            {dcst.NODE_ID},
            {dcst.INCENTIVE},
            {dcst.NETUID},
            {dcst.STAKE},
            {dcst.TRUST},
            {dcst.VTRUST},
            {dcst.LAST_UPDATED},
            {dcst.IP},
            {dcst.IP_TYPE},
            {dcst.PORT},
            {dcst.PROTOCOL}
        )
        VALUES ($1, $2, $3, $4, $5, $6, $7, $8, $9, $10, $11, $12, $13)
        """,
        [
            (
                node.hotkey,
                node.coldkey,
                node.node_id,
                node.incentive,
                node.netuid,
                node.stake,
                node.trust,
                node.vtrust,
                node.last_updated,
                node.ip,
                node.ip_type,
                node.port,
                node.protocol,
            )
            for node in nodes
        ],
    )


async def get_node_by_hotkey(hotkey: str, psql_db: PSQLDB) -> Optional[Node]:
    """Get node by hotkey for the current NETUID"""
    async with await psql_db.connection() as connection:
        connection: Connection
        query = f"""
            SELECT * FROM {dcst.NODES_TABLE}
            WHERE {dcst.HOTKEY} = $1 AND {dcst.NETUID} = $2
        """
        row = await connection.fetchrow(query, hotkey, NETUID)
        if row:
            return Node(**dict(row))
        return None


async def update_our_vali_node_in_db(connection: Connection, ss58_address: str) -> None:
    """Update validator node for the current NETUID"""
    query = f"""
        UPDATE {dcst.NODES_TABLE}
        SET {dcst.OUR_VALIDATOR} = true
        WHERE {dcst.HOTKEY} = $1 AND {dcst.NETUID} = $2
    """
    await connection.execute(query, ss58_address, NETUID)


async def get_vali_ss58_address(psql_db: PSQLDB) -> str | None:
    """Get validator SS58 address for the current NETUID"""
    async with await psql_db.connection() as connection:
        connection: Connection
        query = f"""
            SELECT {dcst.HOTKEY}
            FROM {dcst.NODES_TABLE}
            WHERE {dcst.OUR_VALIDATOR} = true AND {dcst.NETUID} = $1
        """
        row = await connection.fetchrow(query, NETUID)
        if row is None:
            logger.error(f"Cannot find validator node for netuid {NETUID} in the DB. Maybe control node is still syncing?")
            return None
        return row[dcst.HOTKEY]


async def get_last_updated_time_for_nodes(connection: Connection) -> datetime.datetime | None:
    """Get last updated time for nodes in the current NETUID"""
    query = f"""
        SELECT MAX({dcst.CREATED_TIMESTAMP})
        FROM {dcst.NODES_TABLE}
        WHERE {dcst.NETUID} = $1
    """
    return await connection.fetchval(query, NETUID)


async def migrate_nodes_to_history(connection: Connection) -> None:
    """Migrate nodes to history table for the current NETUID"""
    logger.info(f"Migrating nodes to history for NETUID {NETUID}")
    await connection.execute(
        f"""
            INSERT INTO {dcst.NODES_HISTORY_TABLE} (
                {dcst.HOTKEY},
                {dcst.COLDKEY},
                {dcst.INCENTIVE},
                {dcst.NETUID},
                {dcst.STAKE},
                {dcst.TRUST},
                {dcst.VTRUST},
                {dcst.LAST_UPDATED},
                {dcst.IP},
                {dcst.IP_TYPE},
                {dcst.PORT},
                {dcst.PROTOCOL},
                {dcst.NODE_ID}
            )
            SELECT
                {dcst.HOTKEY},
                {dcst.COLDKEY},
                {dcst.INCENTIVE},
                {dcst.NETUID},
                {dcst.STAKE},
                {dcst.TRUST},
                {dcst.VTRUST},
                {dcst.LAST_UPDATED},
                {dcst.IP},
                {dcst.IP_TYPE},
                {dcst.PORT},
                {dcst.PROTOCOL},
                {dcst.NODE_ID}
            FROM {dcst.NODES_TABLE}
            WHERE {dcst.NETUID} = $1
        """,
        NETUID,
    )
    logger.debug(f"Truncating node info table for NETUID {NETUID}")
    await connection.execute(f"DELETE FROM {dcst.NODES_TABLE} WHERE {dcst.NETUID} = $1", NETUID)

    # Get length of nodes table to check if migration was successful
    query = f"""
        SELECT COUNT(*) FROM {dcst.NODES_TABLE}
        WHERE {dcst.NETUID} = $1
    """
    node_entries = await connection.fetchval(query, NETUID)
    logger.debug(f"Node entries: {node_entries}")


async def get_vali_node_id(substrate: SubstrateInterface, ss58_address: str) -> str | None:
    _, uid = query_substrate(substrate, "SubtensorModule", "Uids", [NETUID, ss58_address], return_value=True)
    return uid


async def get_node_id_by_hotkey(hotkey: str, psql_db: PSQLDB) -> int | None:
    """Get node_id by hotkey for the current NETUID"""
    async with await psql_db.connection() as connection:
        connection: Connection
        query = f"""
            SELECT {dcst.NODE_ID} FROM {dcst.NODES_TABLE}
            WHERE {dcst.HOTKEY} = $1 AND {dcst.NETUID} = $2
        """
        return await connection.fetchval(query, hotkey, NETUID)
