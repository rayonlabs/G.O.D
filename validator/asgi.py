import os

import uvicorn
from dotenv import load_dotenv

from validator.utils.util import try_db_connections


load_dotenv(os.getenv("ENV_FILE", ".env"))

from contextlib import asynccontextmanager
from logging import getLogger

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from validator.core.config import load_config
from validator.endpoints.health import factory_router as health_router
from validator.endpoints.tasks import factory_router as tasks_router
from validator.utils.logging import setup_logging


setup_logging()
logger = getLogger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI):
    logger.debug("Entering lifespan context manager")
    config = load_config()

    await try_db_connections(config)

    logger.info("Starting up...")
    app.state.config = config

    yield

    logger.info("Shutting down...")
    await config.psql_db.close()
    await config.redis_db.close()


def factory() -> FastAPI:
    logger.debug("Entering factory function")
    app = FastAPI(lifespan=lifespan)

    app.include_router(health_router())
    app.include_router(tasks_router())

    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    logger.debug(f"App created with {len(app.routes)} routes")
    return app


if __name__ == "__main__":
    logger.info("Starting main validator")

    uvicorn.run(factory(), host="0.0.0.0", port=8010)
