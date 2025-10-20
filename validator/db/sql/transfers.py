"""
Database SQL functions for transfer-related operations
"""

from datetime import datetime
from typing import List
from typing import Optional

from validator.core.transfer_models import ColdkeyBalance
from validator.core.transfer_models import TransferData
from validator.core.transfer_models import TransferProcessingState
from validator.db.database import PSQLDB
from validator.utils.logging import get_logger


logger = get_logger(__name__)


async def insert_transfer(psql_db: PSQLDB, transfer: TransferData) -> bool:
    """
    Insert a new transfer into the database

    Args:
        psql_db: Database connection
        transfer: Transfer data to insert

    Returns:
        bool: True if inserted successfully, False if already exists
    """
    try:
        query = """
        INSERT INTO transfers (
            id, to_ss58, to_hex, from_ss58, from_hex, network, block_number,
            timestamp, amount_rao, fee_rao, transaction_hash, extrinsic_id
        ) VALUES (
            $1, $2, $3, $4, $5, $6, $7, $8, $9, $10, $11, $12
        ) ON CONFLICT (id) DO NOTHING
        RETURNING id
        """

        async with await psql_db.connection() as connection:
            result = await connection.fetchrow(
                query,
                transfer.id,
                transfer.to_ss58,
                transfer.to_hex,
                transfer.from_ss58,
                transfer.from_hex,
                transfer.network,
                transfer.block_number,
                transfer.timestamp,
                transfer.amount_rao,
                transfer.fee_rao,
                transfer.transaction_hash,
                transfer.extrinsic_id,
            )

        return result is not None

    except Exception as e:
        logger.error(f"Failed to insert transfer {transfer.id}: {e}")
        return False


async def get_transfer_processing_state(psql_db: PSQLDB) -> Optional[TransferProcessingState]:
    """
    Get the current transfer processing state

    Args:
        psql_db: Database connection

    Returns:
        TransferProcessingState or None if not found
    """
    try:
        query = """
        SELECT id, last_processed_timestamp, last_processed_block,
               processing_interval_hours, target_address, network,
               created_at, updated_at
        FROM transfer_processing_state
        WHERE id = 1
        """

        async with await psql_db.connection() as connection:
            result = await connection.fetchrow(query)

        if result:
            return TransferProcessingState(
                id=result["id"],
                last_processed_timestamp=result["last_processed_timestamp"],
                last_processed_block=result["last_processed_block"],
                processing_interval_hours=result["processing_interval_hours"],
                target_address=result["target_address"],
                network=result["network"],
                created_at=result["created_at"],
                updated_at=result["updated_at"],
            )

        return None

    except Exception as e:
        logger.error(f"Failed to get transfer processing state: {e}")
        return None


async def update_transfer_processing_state(
    psql_db: PSQLDB,
    last_processed_timestamp: datetime,
    last_processed_block: int,
    target_address: Optional[str] = None,
    network: Optional[str] = None,
    processing_interval_hours: Optional[int] = None,
) -> bool:
    """
    Update the transfer processing state

    Args:
        psql_db: Database connection
        last_processed_timestamp: Last processed timestamp
        last_processed_block: Last processed block number
        target_address: Target address to monitor (optional)
        network: Network to monitor (optional)
        processing_interval_hours: Processing interval in hours (optional)

    Returns:
        bool: True if updated successfully
    """
    try:
        # Build dynamic query based on provided parameters
        set_clauses = ["last_processed_timestamp = $2", "last_processed_block = $3", "updated_at = CURRENT_TIMESTAMP"]
        params = [1, last_processed_timestamp, last_processed_block]
        param_count = 3

        if target_address is not None:
            param_count += 1
            set_clauses.append(f"target_address = ${param_count}")
            params.append(target_address)

        if network is not None:
            param_count += 1
            set_clauses.append(f"network = ${param_count}")
            params.append(network)

        if processing_interval_hours is not None:
            param_count += 1
            set_clauses.append(f"processing_interval_hours = ${param_count}")
            params.append(processing_interval_hours)

        query = f"""
        UPDATE transfer_processing_state
        SET {", ".join(set_clauses)}
        WHERE id = $1
        """

        async with await psql_db.connection() as connection:
            await connection.execute(query, *params)
        return True

    except Exception as e:
        logger.error(f"Failed to update transfer processing state: {e}")
        return False


async def get_or_create_coldkey_balance(psql_db: PSQLDB, coldkey: str) -> Optional[ColdkeyBalance]:
    """
    Get or create a coldkey balance record

    Args:
        psql_db: Database connection
        coldkey: Coldkey SS58 address

    Returns:
        ColdkeyBalance or None if failed
    """
    try:
        # Try to get existing record
        query = """
        SELECT coldkey, balance_rao, total_sent_rao, transfer_count, last_transfer_at, created_at, updated_at
        FROM coldkey_balances
        WHERE coldkey = $1
        """

        async with await psql_db.connection() as connection:
            result = await connection.fetchrow(query, coldkey)

        if result:
            return ColdkeyBalance(
                coldkey=result["coldkey"],
                balance_rao=result["balance_rao"],
                total_sent_rao=result["total_sent_rao"],
                transfer_count=result["transfer_count"],
                last_transfer_at=result["last_transfer_at"],
                created_at=result["created_at"],
                updated_at=result["updated_at"],
            )

        # Create new record if it doesn't exist
        insert_query = """
        INSERT INTO coldkey_balances (coldkey, balance_rao, total_sent_rao, transfer_count)
        VALUES ($1, 0, 0, 0)
        RETURNING coldkey, balance_rao, total_sent_rao, transfer_count, last_transfer_at, created_at, updated_at
        """

        async with await psql_db.connection() as connection:
            result = await connection.fetchrow(insert_query, coldkey)

        if result:
            return ColdkeyBalance(
                coldkey=result["coldkey"],
                balance_rao=result["balance_rao"],
                total_sent_rao=result["total_sent_rao"],
                transfer_count=result["transfer_count"],
                last_transfer_at=result["last_transfer_at"],
                created_at=result["created_at"],
                updated_at=result["updated_at"],
            )

        return None

    except Exception as e:
        logger.error(f"Failed to get or create coldkey balance for {coldkey}: {e}")
        return None


async def update_coldkey_balance(
    psql_db: PSQLDB,
    coldkey: str,
    amount_sent_rao: int,
    transfer_timestamp: datetime,
) -> bool:
    """
    Update a coldkey's balance based on a transfer sent to our target address

    Args:
        psql_db: Database connection
        coldkey: Coldkey SS58 address
        amount_sent_rao: Amount sent to our target address in RAO (always positive)
        transfer_timestamp: Timestamp of the transfer

    Returns:
        bool: True if updated successfully
    """
    try:
        query = """
        UPDATE coldkey_balances
        SET total_sent_rao = total_sent_rao + $2,
            transfer_count = transfer_count + 1,
            last_transfer_at = $3,
            updated_at = CURRENT_TIMESTAMP
        WHERE coldkey = $1
        """

        async with await psql_db.connection() as connection:
            await connection.execute(query, coldkey, amount_sent_rao, transfer_timestamp)
        return True

    except Exception as e:
        logger.error(f"Failed to update coldkey balance for {coldkey}: {e}")
        return False


async def update_coldkey_balance_amount(
    psql_db: PSQLDB,
    coldkey: str,
    new_balance_rao: int,
) -> bool:
    """
    Update a coldkey's current balance amount

    Args:
        psql_db: Database connection
        coldkey: Coldkey SS58 address
        new_balance_rao: New balance amount in RAO

    Returns:
        bool: True if updated successfully
    """
    try:
        query = """
        UPDATE coldkey_balances
        SET balance_rao = $2,
            updated_at = CURRENT_TIMESTAMP
        WHERE coldkey = $1
        """

        async with await psql_db.connection() as connection:
            await connection.execute(query, coldkey, new_balance_rao)
        return True

    except Exception as e:
        logger.error(f"Failed to update coldkey balance amount for {coldkey}: {e}")
        return False


async def get_coldkey_balances(psql_db: PSQLDB, limit: int = 100) -> List[ColdkeyBalance]:
    """
    Get coldkey balances ordered by balance descending

    Args:
        psql_db: Database connection
        limit: Maximum number of records to return

    Returns:
        List of ColdkeyBalance records
    """
    try:
        query = """
        SELECT coldkey, balance_rao, total_sent_rao, transfer_count, last_transfer_at, created_at, updated_at
        FROM coldkey_balances
        ORDER BY balance_rao DESC
        LIMIT $1
        """

        results = await psql_db.fetch_all(query, limit)

        return [
            ColdkeyBalance(
                coldkey=row["coldkey"],
                balance_rao=row["balance_rao"],
                total_sent_rao=row["total_sent_rao"],
                transfer_count=row["transfer_count"],
                last_transfer_at=row["last_transfer_at"],
                created_at=row["created_at"],
                updated_at=row["updated_at"],
            )
            for row in results
        ]

    except Exception as e:
        logger.error(f"Failed to get coldkey balances: {e}")
        return []


async def get_transfers_for_address(
    psql_db: PSQLDB,
    address: str,
    limit: int = 100,
    offset: int = 0,
) -> List[TransferData]:
    """
    Get transfers for a specific address

    Args:
        psql_db: Database connection
        address: Address to search for (can be sender or recipient)
        limit: Maximum number of records to return
        offset: Number of records to skip

    Returns:
        List of TransferData records
    """
    try:
        query = """
        SELECT id, to_ss58, to_hex, from_ss58, from_hex, network, block_number,
               timestamp, amount_rao, fee_rao, transaction_hash, extrinsic_id,
               created_at, updated_at
        FROM transfers
        WHERE to_ss58 = $1 OR from_ss58 = $1
        ORDER BY timestamp DESC
        LIMIT $2 OFFSET $3
        """

        async with await psql_db.connection() as connection:
            results = await connection.fetch(query, address, limit, offset)

        return [
            TransferData(
                id=row["id"],
                to_ss58=row["to_ss58"],
                to_hex=row["to_hex"],
                from_ss58=row["from_ss58"],
                from_hex=row["from_hex"],
                network=row["network"],
                block_number=row["block_number"],
                timestamp=row["timestamp"],
                amount_rao=row["amount_rao"],
                fee_rao=row["fee_rao"],
                transaction_hash=row["transaction_hash"],
                extrinsic_id=row["extrinsic_id"],
                created_at=row["created_at"],
                updated_at=row["updated_at"],
            )
            for row in results
        ]

    except Exception as e:
        logger.error(f"Failed to get transfers for address {address}: {e}")
        return []


async def get_coldkey_balance_by_address(psql_db: PSQLDB, coldkey: str) -> Optional[ColdkeyBalance]:
    """
    Get balance information for a specific coldkey address

    Args:
        psql_db: Database connection
        coldkey: Coldkey SS58 address

    Returns:
        ColdkeyBalance or None if not found
    """
    try:
        query = """
        SELECT coldkey, balance_rao, total_sent_rao, transfer_count, last_transfer_at, created_at, updated_at
        FROM coldkey_balances
        WHERE coldkey = $1
        """

        async with await psql_db.connection() as connection:
            result = await connection.fetchrow(query, coldkey)

        if result:
            return ColdkeyBalance(
                coldkey=result["coldkey"],
                balance_rao=result["balance_rao"],
                total_sent_rao=result["total_sent_rao"],
                transfer_count=result["transfer_count"],
                last_transfer_at=result["last_transfer_at"],
                created_at=result["created_at"],
                updated_at=result["updated_at"],
            )
        return None

    except Exception as e:
        logger.error(f"Failed to get coldkey balance for {coldkey}: {e}")
        return None
