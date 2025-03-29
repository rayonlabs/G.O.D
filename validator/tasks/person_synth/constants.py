import os

FACE_IMAGE_PATH = "validator/tasks/person_synth/ComfyUI/input/person.jpg"
FACE_IMAGE_URL = "https://thispersondoesnotexist.com/"
LLAVA_MODEL_PATH = "validator/tasks/person_synth/cache/llava-v1.5-7b"
WORKFLOW_PATH = "validator/tasks/person_synth/person_avatars_template.json"
SAFETY_CHECKER_MODEL_PATH = "validator/tasks/person_synth/cache/Juggernaut_final"
DEFAULT_SAVE_DIR = "/app/avatars/"


#prompt stuff
NUM_PROMPTS = int(os.getenv("NUM_PROMPTS", 15))
PERSON_PROMPT = f"""
        Here is an image of a person. Generate {NUM_PROMPTS} different prompts for creating an avatar of the person.
        Place them in different places, backgrounds, scenarios, and emotions.
        Use different settings like beach, house, room, park, office, city, and others.
        Also use a different range of emotions like happy, sad, smiling, laughing, angry, thinking for every prompt.
    """