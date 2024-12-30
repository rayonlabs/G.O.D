from validator.core.config import Config
from validator.utils.logging import get_logger


logger = get_logger(__name__)


async def try_db_connections(config: Config) -> None:
    logger.info("Attempting to connect to PostgreSQL...")
    await config.psql_db.connect()
    await config.psql_db.pool.execute("SELECT 1=1 as one")
    logger.info("PostgreSQL connected successfully")

    logger.info("Attempting to connect to Redis")
    await config.redis_db.ping()
    logger.info("Redis connected successfully")
