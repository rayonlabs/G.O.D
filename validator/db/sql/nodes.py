import datetime

from asyncpg.connection import Connection
from fiber import SubstrateInterface
from fiber.chain.models import Node

from core.constants import NETUID
from validator.db import constants as dcst
from validator.db.database import PSQLDB
from validator.utils.logging import get_logger
from validator.utils.query_substrate import query_substrate


logger = get_logger(__name__)


async def get_eligible_nodes(psql_db: PSQLDB) -> list[Node]:
    """
    Get all nodes eligible for tasks.

    Includes nodes that either:
    a) Do not have any entries in the task_nodes table (new nodes with no scores)
    b) Have at least one positive quality_score within the last 7 days
    c) Have entries but all scores are NULL (not yet evaluated)
    """
    logger.info("Getting eligible nodes (new nodes, nodes with NULL scores, or nodes with positive scores in the last 7 days)")
    async with await psql_db.connection() as connection:
        connection: Connection
        query = f"""
            SELECT n.* FROM {dcst.NODES_TABLE} n
            WHERE n.{dcst.NETUID} = $1
            AND (
                -- Condition a: No entries in task_nodes table
                NOT EXISTS (
                    SELECT 1 FROM {dcst.TASK_NODES_TABLE} tn
                    WHERE tn.{dcst.HOTKEY} = n.{dcst.HOTKEY}
                )
                OR
                -- Condition b: At least one positive quality_score within the last 7 days
                EXISTS (
                    SELECT 1 FROM {dcst.TASK_NODES_TABLE} tn
                    JOIN {dcst.TASKS_TABLE} t ON tn.{dcst.TASK_ID} = t.{dcst.TASK_ID}
                    WHERE tn.{dcst.HOTKEY} = n.{dcst.HOTKEY}
                    AND tn.{dcst.TASK_NODE_QUALITY_SCORE} > 0
                    AND t.{dcst.CREATED_AT} >= NOW() - INTERVAL '7 days'
                )
                OR
                -- Condition c: Has entries but all scores are NULL
                (
                    EXISTS (
                        SELECT 1 FROM {dcst.TASK_NODES_TABLE} tn
                        WHERE tn.{dcst.HOTKEY} = n.{dcst.HOTKEY}
                    )
                    AND NOT EXISTS (
                        SELECT 1 FROM {dcst.TASK_NODES_TABLE} tn
                        WHERE tn.{dcst.HOTKEY} = n.{dcst.HOTKEY}
                        AND tn.{dcst.TASK_NODE_QUALITY_SCORE} IS NOT NULL
                    )
                )
            )
        """
        rows = await connection.fetch(query, NETUID)
        eligible_nodes = [Node(**dict(row)) for row in rows]
        logger.info(f"Found {len(eligible_nodes)} eligible nodes")
        return eligible_nodes


async def get_all_nodes(psql_db: PSQLDB) -> list[Node]:
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
            {dcst.ALPHA_STAKE},
            {dcst.TAO_STAKE},
            {dcst.STAKE},
            {dcst.TRUST},
            {dcst.VTRUST},
            {dcst.LAST_UPDATED},
            {dcst.IP},
            {dcst.IP_TYPE},
            {dcst.PORT},
            {dcst.PROTOCOL}
        )
        VALUES ($1, $2, $3, $4, $5, $6, $7, $8, $9, $10, $11, $12, $13, $14, $15)
        """,
        [
            (
                node.hotkey,
                node.coldkey,
                node.node_id,
                node.incentive,
                node.netuid,
                node.alpha_stake,
                node.tao_stake,
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


async def get_node_by_hotkey(hotkey: str, psql_db: PSQLDB) -> Node | None:
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
                {dcst.ALPHA_STAKE},
                {dcst.TAO_STAKE},
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
                {dcst.ALPHA_STAKE},
                {dcst.TAO_STAKE},
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
