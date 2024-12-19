import os

import uvicorn
from dotenv import load_dotenv

from validator.utils.util import try_db_connections


load_dotenv(os.getenv("ENV_FILE", ".env"))

from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fiber.logging_utils import get_logger

from validator.core.config import load_config
from validator.endpoints.health import factory_router as health_router
from validator.endpoints.tasks import factory_router as tasks_router
from scalar_fastapi import get_scalar_api_reference
from opentelemetry.instrumentation.fastapi import FastAPIInstrumentor


logger = get_logger(__name__)


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

    app.add_api_route(
        "/scalar",
        lambda: get_scalar_api_reference(openapi_url=app.openapi_url, title=app.title),
        methods=["GET"],
    )

    app.include_router(health_router())
    app.include_router(tasks_router())

    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )
    
    FastAPIInstrumentor().instrument_app(app)

    logger.debug(f"App created with {len(app.routes)} routes")
    return app


if __name__ == "__main__":
    logger.info("Starting main validator")

    uvicorn.run(factory(), host="0.0.0.0", port=8010)
