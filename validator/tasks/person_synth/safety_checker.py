import numpy as np
from PIL import Image
from functools import partial
import torch
from torch import nn
from diffusers import StableDiffusionPipeline
from transformers import CLIPFeatureExtractor
import os

import validator.tasks.person_synth.constants as cst

def cosine_distance(image_embeds, text_embeds):
    normalized_image_embeds = nn.functional.normalize(image_embeds)
    normalized_text_embeds = nn.functional.normalize(text_embeds)
    return torch.mm(normalized_image_embeds, normalized_text_embeds.t())

@torch.no_grad()
def forward_inspect(self, clip_input, images):
    pooled_output = self.vision_model(clip_input)[1]
    image_embeds = self.visual_projection(pooled_output)

    special_cos_dist = cosine_distance(image_embeds, self.special_care_embeds).cpu().numpy()
    cos_dist = cosine_distance(image_embeds, self.concept_embeds).cpu().numpy()

    matches = {"nsfw": [], "special": []}
    batch_size = image_embeds.shape[0]
    for i in range(batch_size):
        result_img = {
            "special_scores": {},
            "special_care": [],
            "concept_scores": {},
            "bad_concepts": [],
        }

        adjustment = -0.01

        for concet_idx in range(len(special_cos_dist[0])):
            concept_cos = special_cos_dist[i][concet_idx]
            concept_threshold = self.special_care_embeds_weights[concet_idx].item()
            result_img["special_scores"][concet_idx] = round(concept_cos - concept_threshold + adjustment, 3)
            if result_img["special_scores"][concet_idx] > 0:
                result_img["special_care"].append({concet_idx, result_img["special_scores"][concet_idx]})

        for concet_idx in range(len(cos_dist[0])):
            concept_cos = cos_dist[i][concet_idx]
            concept_threshold = self.concept_embeds_weights[concet_idx].item()
            result_img["concept_scores"][concet_idx] = round(concept_cos - concept_threshold + adjustment, 3)

            if result_img["concept_scores"][concet_idx] > 0:
                result_img["bad_concepts"].append(concet_idx)

    has_nsfw_concepts = len(matches["nsfw"]) > 0

    return matches, has_nsfw_concepts

device = os.getenv("DEVICE", "cuda:0")
device = device if "cuda" in device else f"cuda:{device}"
safety_pipe = StableDiffusionPipeline.from_pretrained(cst.SAFETY_CHECKER_MODEL_PATH, torch_dtype=torch.bfloat16, local_files_only=True).to(device)
safety_pipe.safety_checker.forward = partial(forward_inspect, self=safety_pipe.safety_checker)
safety_feature_extractor = CLIPFeatureExtractor.from_pretrained(f"{cst.SAFETY_CHECKER_MODEL_PATH}/feature_extractor", local_files_only=True)
safety_checker = safety_pipe.safety_checker


def nsfw_check(image: Image.Image) -> bool:
    image_np = np.array(image)
    if np.all(image_np == 0):
        return True
    with torch.cuda.amp.autocast():
        safety_checker_input = safety_feature_extractor(images=image, return_tensors="pt").to(device)
        result, has_nsfw_concepts = safety_checker.forward(clip_input=safety_checker_input.pixel_values, images=image)
    return bool(has_nsfw_concepts)