"""
Docker utilities for building and pushing images to DockerHub
"""
import asyncio
import os
import uuid

import docker
from docker.errors import APIError
from docker.errors import BuildError

from trainer import constants as cst
from trainer.utils.logging import logger
from validator.utils.logging import stream_image_build_logs


async def build_and_push_image(
    dockerfile_path: str,
    context_path: str,
    dockerhub_username: str,
    dockerhub_password: str,
    is_image_task: bool = False,
    tag_suffix: str | None = None,
    log_labels: dict[str, str] | None = None,
    no_cache: bool = True,
) -> tuple[str | None, str | None]:
    """
    Build Docker image and push to DockerHub.
    
    Args:
        dockerfile_path: Path to Dockerfile
        context_path: Build context path
        dockerhub_username: DockerHub username
        dockerhub_password: DockerHub password/token
        is_image_task: Whether this is an image training task
        tag_suffix: Optional suffix for the tag (defaults to UUID)
        log_labels: Labels for logging
        no_cache: Whether to build without cache
        
    Returns:
        tuple: (image_tag, error_message)
    """
    client: docker.DockerClient = docker.from_env()
    
    # Generate tag
    if tag_suffix:
        tag_base = f"{dockerhub_username}/trainer-{'image' if is_image_task else 'text'}-{tag_suffix}"
    else:
        tag_base = f"{dockerhub_username}/trainer-{'image' if is_image_task else 'text'}-{uuid.uuid4().hex[:8]}"
    
    tag = f"{tag_base}:latest"
    
    logger.info(f"Building Docker image '{tag}', Dockerfile: {dockerfile_path}, Context: {context_path}...", extra=log_labels)
    
    try:
        # Build image
        build_output = client.api.build(
            path=context_path,
            dockerfile=dockerfile_path,
            tag=tag,
            nocache=no_cache,
            decode=True,
        )
        stream_image_build_logs(build_output, logger=logger, log_context=log_labels)
        
        logger.info(f"Docker image built successfully: {tag}", extra=log_labels)
        
        # Login to DockerHub
        logger.info("Logging into DockerHub...", extra=log_labels)
        await asyncio.to_thread(
            client.login,
            username=dockerhub_username,
            password=dockerhub_password,
        )
        
        # Push image
        logger.info(f"Pushing image to DockerHub: {tag}...", extra=log_labels)
        push_output = client.images.push(
            repository=tag_base,
            tag="latest",
            stream=True,
            decode=True,
        )
        
        # Stream push logs
        for line in push_output:
            if "error" in line.get("status", "").lower() or "errorDetail" in line:
                error_msg = line.get("error", line.get("errorDetail", {}).get("message", "Unknown error"))
                logger.error(f"DockerHub push error: {error_msg}", extra=log_labels)
                return None, error_msg
            if "status" in line:
                logger.debug(f"Push: {line.get('status', '')}", extra=log_labels)
        
        logger.info(f"Image pushed successfully to DockerHub: {tag}", extra=log_labels)
        return tag, None
        
    except (BuildError, APIError) as e:
        error_msg = str(e)
        logger.error(f"Docker build/push failed: {error_msg}", extra=log_labels)
        return None, error_msg
    except Exception as e:
        error_msg = str(e)
        logger.error(f"Unexpected error during build/push: {error_msg}", extra=log_labels)
        return None, error_msg

