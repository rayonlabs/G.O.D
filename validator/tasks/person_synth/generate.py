import requests
import uuid
import re
import json
from PIL import Image
import random
import os
from copy import deepcopy
import io
from io import BytesIO
from contextlib import redirect_stdout

from llava.mm_utils import get_model_name_from_path
from llava.eval.run_llava import eval_model


import validator.utils.comfy_api_gate as api_gate
import validator.tasks.person_synth.constants as cst
from validator.tasks.person_synth.safety_checker import Safety_Checker


sc = Safety_Checker()

with open(cst.WORKFLOW_PATH, "r") as file:
    avatar_template = json.load(file)

def get_face_image():
    response = requests.get("https://thispersondoesnotexist.com/")
    response.raise_for_status()
    image = Image.open(BytesIO(response.content))
    return image

if __name__ == "__main__":
    face_image = get_face_image()
    face_image.save(cst.FACE_IMAGE_PATH)
    num_prompts = random.randint(11,20)

    prompt = f"""
        Here is an image of a person. Generate {num_prompts} different prompts for creating an avatar of the person.
        Place them in different places, backgrounds, scenarios, and emotions.
        Use different settings like beach, house, room, park, office, city, and others.
        Also use a different range of emotions like happy, sad, smiling, laughing, angry, thinking for every prompt.
    """

    args = type('Args', (), {
        "model_path": cst.LLAVA_MODEL_PATH,
        "model_base": None,
        "model_name": get_model_name_from_path(cst.LLAVA_MODEL_PATH),
        "query": prompt,
        "conv_mode": None,
        "image_file": cst.FACE_IMAGE_PATH,
        "sep": ",",
        "temperature": 0.8,
        "top_p": None,
        "num_beams": 1,
        "max_new_tokens": 6000
    })()


    f = io.StringIO()
    with redirect_stdout(f):
        eval_model(args)
    output = f.getvalue()
    prompts = re.findall(r"\d+\.\s(.+)", str(output), re.MULTILINE)

    api_gate.connect()
    save_dir = os.getenv("SAVE_DIR", "/app/avatars/")

    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    for prompt in prompts:
        workflow = deepcopy(avatar_template)
        workflow["Prompt"]["inputs"]["text"] += prompt
        image = api_gate.generate(workflow)[0]
        if not sc.nsfw_check(image):
            image_id = uuid.uuid4()
            image.save(f"{save_dir}{image_id}.png")
            with open(f"{save_dir}{image_id}.txt", "w") as file:
                file.write(prompt)
   

