import base64
import json
import os
import random
import re
import tempfile
import uuid
import shutil
from datetime import datetime
from datetime import timedelta
from pathlib import Path
from typing import AsyncGenerator
from typing import List
import asyncio
import docker
from docker.models.containers import Container

from fiber import Keypair

import validator.core.constants as cst
from core.models.payload_models import ImageTextPair
from core.models.utility_models import Message
from core.models.utility_models import Role
from core.models.utility_models import TaskStatus
from validator.core.config import Config
from validator.core.models import ImageRawTask
from validator.core.models import RawTask
from validator.db.sql.tasks import add_task
from validator.tasks.task_prep import upload_file_to_minio
from validator.utils.call_endpoint import post_to_nineteen_chat
from validator.utils.call_endpoint import post_to_nineteen_image
from validator.utils.call_endpoint import retry_with_backoff
from validator.utils.llm import convert_to_nineteen_payload
from validator.utils.logging import get_logger


logger = get_logger(__name__)

IMAGE_STYLES = [
    "Watercolor Painting",
    "Oil Painting",
    "Digital Art",
    "Pencil Sketch",
    "Comic Book Style",
    "Cyberpunk",
    "Steampunk",
    "Impressionist",
    "Pop Art",
    "Minimalist",
    "Gothic",
    "Art Nouveau",
    "Pixel Art",
    "Anime",
    "3D Render",
    "Low Poly",
    "Photorealistic",
    "Vector Art",
    "Abstract Expressionism",
    "Realism",
    "Futurism",
    "Cubism",
    "Surrealism",
    "Baroque",
    "Renaissance",
    "Fantasy Illustration",
    "Sci-Fi Illustration",
    "Ukiyo-e",
    "Line Art",
    "Black and White Ink Drawing",
    "Graffiti Art",
    "Stencil Art",
    "Flat Design",
    "Isometric Art",
    "Retro 80s Style",
    "Vaporwave",
    "Dreamlike",
    "High Fantasy",
    "Dark Fantasy",
    "Medieval Art",
    "Art Deco",
    "Hyperrealism",
    "Sculpture Art",
    "Caricature",
    "Chibi",
    "Noir Style",
    "Lowbrow Art",
    "Psychedelic Art",
    "Vintage Poster",
    "Manga",
    "Holographic",
    "Kawaii",
    "Monochrome",
    "Geometric Art",
    "Photocollage",
    "Mixed Media",
    "Ink Wash Painting",
    "Charcoal Drawing",
    "Concept Art",
    "Digital Matte Painting",
    "Pointillism",
    "Expressionism",
    "Sumi-e",
    "Retro Futurism",
    "Pixelated Glitch Art",
    "Neon Glow",
    "Street Art",
    "Acrylic Painting",
    "Bauhaus",
    "Flat Cartoon Style",
    "Carved Relief Art",
    "Fantasy Realism",
]


def create_diffusion_messages(style: str, num_prompts: int) -> List[Message]:
    system_content = """You are an expert in creating diverse and descriptive prompts for image generation models.
Your task is to generate detailed and creative prompts that strongly embody specific styles.
You will return the prompts in a JSON format with no additional text.

Example Output:
{
  "prompts": [
    "A pixel art scene of a medieval castle with knights guarding the entrance, surrounded by a moat",
    "A pixel art depiction of a bustling futuristic city with flying cars zooming past neon-lit skyscrapers"
  ]
}"""

    user_content = f"""Generate {num_prompts} prompts in {style} style.

Requirements:
- Each prompt must clearly communicate the {style}'s distinctive visual characteristics
- Include specific visual elements that define this style (textures, colors, techniques)
- Vary subject matter and get creative! 
- Remembering to return JSON only"""

    return [Message(role=Role.SYSTEM, content=system_content), Message(role=Role.USER, content=user_content)]


@retry_with_backoff
async def generate_diffusion_prompts(style: str, keypair: Keypair, num_prompts: int) -> List[str]:
    messages = create_diffusion_messages(style, num_prompts)
    payload = convert_to_nineteen_payload(messages, cst.IMAGE_PROMPT_GEN_MODEL, cst.IMAGE_PROMPT_GEN_MODEL_TEMPERATURE)

    result = await post_to_nineteen_chat(payload, keypair)

    try:
        if isinstance(result, str):
            json_match = re.search(r"\{[\s\S]*\}", result)
            if json_match:
                result = json_match.group(0)
            else:
                raise ValueError("Failed to generate a valid json")

        result_dict = json.loads(result) if isinstance(result, str) else result
        return result_dict["prompts"]
    except (json.JSONDecodeError, KeyError) as e:
        raise ValueError(f"Failed to generate valid diffusion prompts: {e}")


@retry_with_backoff
async def generate_image(prompt: str, keypair: Keypair, width: int, height: int) -> str:
    """Generate an image using the Nineteen AI API.

    Args:
        prompt: The text prompt to generate an image from
        keypair: The keypair containing the API key
        width: The width in pixels of the image to generate
        height: The height in pixels of the image to generate

    Returns:
        str: The base64-encoded image data
    """
    payload = {
        "prompt": prompt,
        "model": cst.IMAGE_GEN_MODEL,
        "steps": cst.IMAGE_GEN_STEPS,
        "cfg_scale": cst.IMAGE_GEN_CFG_SCALE,
        "height": height,
        "width": width,
        "negative_prompt": "",
    }

    result = await post_to_nineteen_image(payload, keypair)

    try:
        result_dict = json.loads(result) if isinstance(result, str) else result
        return result_dict["image_b64"]
    except (json.JSONDecodeError, KeyError) as e:
        logger.error(f"Error parsing image generation response: {e}")
        raise ValueError("Failed to generate image")

async def generate_style_synthetic(config: Config, num_prompts: int) -> tuple[list[ImageTextPair], str]:
    style = random.choice(IMAGE_STYLES)
    try:
        prompts = await generate_diffusion_prompts(style, config.keypair, num_prompts)
    except Exception as e:
        logger.error(f"Failed to generate prompts for {style}: {e}")
        raise e
    image_text_pairs = []
    for i, prompt in enumerate(prompts):
        width = random.randrange(cst.MIN_IMAGE_WIDTH, cst.MAX_IMAGE_WIDTH + 1, cst.IMAGE_RESOLUTION_STEP)
        height = random.randrange(cst.MIN_IMAGE_HEIGHT, cst.MAX_IMAGE_HEIGHT + 1, cst.IMAGE_RESOLUTION_STEP)
        image = await generate_image(prompt, config.keypair, width, height)

        with tempfile.NamedTemporaryFile(dir=cst.TEMP_PATH_FOR_IMAGES, suffix=".png") as img_file:
            img_file.write(base64.b64decode(image))
            img_url = await upload_file_to_minio(img_file.name, cst.BUCKET_NAME, f"{os.urandom(8).hex()}_{i}.png")

        with tempfile.NamedTemporaryFile(suffix=".txt") as txt_file:
            txt_file.write(prompt.encode())
            txt_file.flush()
            txt_file.seek(0)
            txt_url = await upload_file_to_minio(txt_file.name, cst.BUCKET_NAME, f"{os.urandom(8).hex()}_{i}.txt")

        image_text_pairs.append(ImageTextPair(image_url=img_url, text_url=txt_url))

    return image_text_pairs, style

async def generate_person_synthetic(num_prompts: int) -> tuple[list[ImageTextPair], str]:
    client = docker.from_env()
    image_text_pairs = []
    with tempfile.TemporaryDirectory(dir=cst.TEMP_PATH_FOR_IMAGES) as tmp_dir_path:
        container = await asyncio.to_thread(
            client.containers.run,
            image=cst.PERSON_SYNTH_DOCKER_IMAGE,
            environment={"SAVE_DIR": cst.PERSON_SYNTH_CONTAINER_SAVE_PATH, "NUM_PROMPTS": num_prompts},
            volumes={tmp_dir_path: {"bind": cst.PERSON_SYNTH_CONTAINER_SAVE_PATH, "mode": "rw"}},
            device_requests=[docker.types.DeviceRequest(capabilities=[["gpu"]], device_ids=["0"])],
            detach=True
        )
        result = await asyncio.to_thread(container.wait)
        images_dir = Path(tmp_dir_path)
        for file in images_dir.iterdir():
            if file.is_file():
                if file.suffix == ".png":
                    img_url = await upload_file_to_minio(f"{tmp_dir_path}/{file.name}", cst.BUCKET_NAME, f"{os.urandom(8).hex()}.png")
                    txt_url = await upload_file_to_minio(f"{tmp_dir_path}/{file.stem}.txt", cst.BUCKET_NAME, f"{os.urandom(8).hex()}.txt")
                    image_text_pairs.append(ImageTextPair(image_url=img_url, text_url=txt_url))
        if os.path.exists(tmp_dir_path):
            shutil.rmtree(tmp_dir_path)


    await asyncio.to_thread(client.containers.prune)
    await asyncio.to_thread(client.images.prune, filters={"dangling": True})
    await asyncio.to_thread(client.volumes.prune) 

    return image_text_pairs, cst.PERSON_SYNTH_DS_PREFIX

async def create_synthetic_image_task(config: Config, models: AsyncGenerator[str, None]) -> RawTask:
    """Create a synthetic image task with random model and style."""
    number_of_hours = random.randint(cst.MIN_IMAGE_COMPETITION_HOURS, cst.MAX_IMAGE_COMPETITION_HOURS)
    num_prompts = random.randint(cst.MIN_IMAGE_SYNTH_PAIRS, cst.MAX_IMAGE_SYNTH_PAIRS)
    model_id = await anext(models)
    Path(cst.TEMP_PATH_FOR_IMAGES).mkdir(parents=True, exist_ok=True)
    if random.random() < cst.PERCENTAGE_OF_IMAGE_SYNTHS_SHOULD_BE_STYLE:
        image_text_pairs, ds_prefix = await generate_style_synthetic(config, num_prompts)
    else:
        image_text_pairs, ds_prefix = await generate_person_synthetic(num_prompts)

    task = ImageRawTask(
        model_id=model_id,
        ds=ds_prefix.replace(" ", "_").lower() + "_" + str(uuid.uuid4()),
        image_text_pairs=image_text_pairs,
        status=TaskStatus.PENDING,
        is_organic=False,
        created_at=datetime.utcnow(),
        termination_at=datetime.utcnow() + timedelta(hours=number_of_hours),
        hours_to_complete=number_of_hours,
        account_id=cst.NULL_ACCOUNT_ID,
    )

    logger.info(f"New task created and added to the queue {task}")
    task = await add_task(task, config.psql_db)
    return task
