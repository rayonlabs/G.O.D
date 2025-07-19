import uvicorn
from dotenv import load_dotenv
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from trainer.endpoints import factory_router
from trainer.tasks import start_cleanup_loop_in_thread
from validator.utils.logging import get_logger


load_dotenv(".trainer.env")

logger = get_logger(__name__)


def factory() -> FastAPI:
    logger.debug("Entering factory function")
    app = FastAPI()
    app.include_router(factory_router())
    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    @app.on_event("startup")
    async def startup():
        logger.info("Starting async cleanup loop in a background thread")
        start_cleanup_loop_in_thread()

    return app


app = factory()

if __name__ == "__main__":
    logger.info("Starting trainer")
    uvicorn.run(app, host="0.0.0.0", port=8001)
