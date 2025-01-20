import base64
import json
import random
import re
import shutil
import tempfile
import time
from datetime import datetime
from datetime import timedelta
from pathlib import Path
from typing import List

from fiber import Keypair

import validator.core.constants as cst
from core.models.utility_models import Message
from core.models.utility_models import Role
from core.models.utility_models import TaskStatus
from validator.core.config import Config
from validator.core.constants import SYNTH_MODEL
from validator.core.constants import SYNTH_MODEL_TEMPERATURE
from validator.core.models import ImageRawTask
from validator.core.models import RawTask
from validator.db.sql.tasks import add_task
from validator.tasks.task_prep import upload_file_to_minio
from validator.utils.call_endpoint import post_to_nineteen_chat
from validator.utils.call_endpoint import post_to_nineteen_image
from validator.utils.call_endpoint import retry_with_backoff
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

IMAGE_MODEL_TO_FINETUNE = "stabilityai/stable-diffusion-xl-base-1.0"


def create_diffusion_messages(style: str) -> List[Message]:
    system_content = """You are an expert in creating diverse and descriptive prompts for image generation models.
You will generate a set of creative prompts in a specific artistic style.
Each prompt should be detailed and consistent with the given style.
You will return the prompts in a JSON format with no additional text.

Example Output:
{
  "prompts": [
    "A pixel art scene of a medieval castle with knights guarding the entrance, surrounded by a moat",
    "A pixel art depiction of a bustling futuristic city with flying cars zooming past neon-lit skyscrapers"
  ]
}"""

    user_content = f"""Generate 10 creative and detailed prompts in the following style: {style}
Make sure each prompt is descriptive and would work well with image generation models.
Return only the JSON response."""

    return [Message(role=Role.SYSTEM, content=system_content), Message(role=Role.USER, content=user_content)]


def convert_to_nineteen_payload(
    messages: List[Message], model: str = SYNTH_MODEL, temperature: float = SYNTH_MODEL_TEMPERATURE, stream: bool = False
) -> dict:
    return {
        "messages": [message.model_dump() for message in messages],
        "model": model,
        "temperature": temperature,
        "stream": stream,
    }


@retry_with_backoff
async def generate_diffusion_prompts(style: str, keypair: Keypair) -> List[str]:
    messages = create_diffusion_messages(style)
    payload = convert_to_nineteen_payload(messages)

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
async def generate_image(prompt: str, keypair: Keypair) -> str:
    """Generate an image using the Nineteen AI API.

    Args:
        prompt: The text prompt to generate an image from
        keypair: The keypair containing the API key

    Returns:
        str: The base64-encoded image data
    """
    payload = {
        "prompt": prompt,
        "model": cst.IMAGE_GEN_MODEL,
        "steps": cst.IMAGE_GEN_STEPS,
        "cfg_scale": cst.IMAGE_GEN_CFG_SCALE,
        "height": cst.IMAGE_GEN_HEIGHT,
        "width": cst.IMAGE_GEN_WIDTH,
        "negative_prompt": "",
    }

    result = await post_to_nineteen_image(payload, keypair)

    try:
        result_dict = json.loads(result) if isinstance(result, str) else result
        return result_dict["image_b64"]
    except (json.JSONDecodeError, KeyError) as e:
        logger.error(f"Error parsing image generation response: {e}")
        raise ValueError("Failed to generate image")


async def create_training_data_zip(config: Config, prompts: List[str], style: str, temp_dir: str) -> str:
    source_dir = Path(temp_dir) / "source"
    source_dir.mkdir(exist_ok=True)

    zip_base_path = Path(temp_dir) / f"ds_{len(prompts)}_{style}_{int(time.time())}"

    for i, prompt in enumerate(prompts):
        image = await generate_image(prompt, config.keypair)
        logger.info(f"Generated synthetic image {i+1}/{len(prompts)}")
        with open(source_dir / f"{i}.png", "wb") as f:
            f.write(base64.b64decode(image))
        with open(source_dir / f"{i}.txt", "w") as f:
            f.write(prompt)

    zip_path = shutil.make_archive(zip_base_path, "zip", source_dir)

    return zip_path


async def create_synthetic_image_task(config: Config) -> RawTask:
    number_of_hours = random.randint(cst.MIN_IMAGE_COMPETITION_HOURS, cst.MAX_IMAGE_COMPETITION_HOURS)
    style = random.choice(IMAGE_STYLES)
    try:
        prompts = await generate_diffusion_prompts(style, config.keypair)
    except Exception as e:
        logger.error(f"Failed to generate prompts for {style}: {e}")
        raise e

    Path(cst.TEMP_PATH_FOR_IMAGES).mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(dir=cst.TEMP_PATH_FOR_IMAGES, prefix="diffusion_synth_") as temp_dir:
        zip_path = await create_training_data_zip(config, prompts, style, temp_dir)

        s3_url = await upload_file_to_minio(zip_path, cst.BUCKET_NAME, Path(zip_path).name)

    if s3_url is None:
        raise ValueError("Failed to upload file to MinIO")

    task = ImageRawTask(
        model_id=IMAGE_MODEL_TO_FINETUNE,
        ds=s3_url,
        status=TaskStatus.PENDING,
        is_organic=False,
        created_at=datetime.utcnow(),
        termination_at=datetime.utcnow() + timedelta(hours=number_of_hours),
        hours_to_complete=number_of_hours,
        account_id=cst.NULL_ACCOUNT_ID,
        model_filename="test",
    )

    logger.info(f"New task created and added to the queue {task}")
    task = await add_task(task, config.psql_db)
    return task
