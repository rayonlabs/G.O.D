import uuid
from datetime import datetime
from enum import Enum
from uuid import UUID

from pydantic import BaseModel
from pydantic import ConfigDict
from pydantic import Field


class FileFormat(str, Enum):
    CSV = "csv"  # needs to be local file
    JSON = "json"  # needs to be local file
    HF = "hf"  # Hugging Face dataset
    S3 = "s3"


class JobStatus(str, Enum):
    QUEUED = "Queued"
    RUNNING = "Running"
    COMPLETED = "Completed"
    FAILED = "Failed"
    NOT_FOUND = "Not Found"


class TaskStatus(str, Enum):
    PENDING = "pending"
    PREPARING_DATA = "preparing_data"
    PREP_TASK_FAILURE = "prep_task_failure"
    LOOKING_FOR_NODES = "looking_for_nodes"
    FAILURE_FINDING_NODES = "failure_finding_nodes"
    DELAYED = "delayed"
    READY = "ready"
    TRAINING = "training"
    PREEVALUATION = "preevaluation"
    EVALUATING = "evaluating"
    SUCCESS = "success"
    FAILURE = "failure"


class WinningSubmission(BaseModel):
    hotkey: str
    score: float
    model_repo: str

    # Turn off protected namespace for model
    model_config = ConfigDict(protected_namespaces=())


class MinerSubmission(BaseModel):
    repo: str
    model_hash: str | None = None


class MinerTaskResult(BaseModel):
    hotkey: str
    quality_score: float
    test_loss: float | None
    synth_loss: float | None
    score_reason: str | None


# NOTE: Confusing name with the class above
class TaskMinerResult(BaseModel):
    task_id: UUID
    quality_score: float


class InstructTextDatasetType(BaseModel):
    system_prompt: str | None = ""
    system_format: str | None = "{system}"
    field_system: str | None = None
    field_instruction: str | None = None
    field_input: str | None = None
    field_output: str | None = None
    format: str | None = None
    no_input_format: str | None = None
    field: str | None = None


class RewardFunction(BaseModel):
    """Model representing a reward function with its metadata"""

    reward_id: str | None = Field(None, description="UUID of the reward function in the database")
    reward_func: str = Field(
        ...,
        description="String with the python code of the reward function to use",
        examples=[
            "def reward_func_conciseness(completions, **kwargs):",
            '"""Reward function that favors shorter, more concise answers."""',
            "    return [100.0/(len(completion.split()) + 10) for completion in completions]",
        ],
    )
    reward_weight: float = Field(..., ge=0)
    func_hash: str | None = None
    is_generic: bool | None = None
    is_manual: bool | None = None


class GrpoDatasetType(BaseModel):
    field_prompt: str | None = None
    reward_functions: list[RewardFunction] | None = []
    extra_column: str | None = None


class DpoDatasetType(BaseModel):
    field_prompt: str | None = None
    field_system: str | None = None
    field_chosen: str | None = None
    field_rejected: str | None = None
    prompt_format: str | None = "{prompt}"
    chosen_format: str | None = "{chosen}"
    rejected_format: str | None = "{rejected}"


class ChatTemplateDatasetType(BaseModel):
    chat_template: str | None = "chatml"
    chat_column: str | None = "conversations"
    chat_role_field: str | None = "from"
    chat_content_field: str | None = "value"
    chat_user_reference: str | None = "user"
    chat_assistant_reference: str | None = "assistant"


class ImageModelType(str, Enum):
    FLUX = "flux"
    SDXL = "sdxl"


class Job(BaseModel):
    job_id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    model: str
    status: JobStatus = JobStatus.QUEUED
    error_message: str | None = None
    expected_repo_name: str | None = None


TextDatasetType = InstructTextDatasetType | DpoDatasetType | GrpoDatasetType | ChatTemplateDatasetType


class TextJob(Job):
    dataset: str
    dataset_type: TextDatasetType
    file_format: FileFormat


class DiffusionJob(Job):
    model_config = ConfigDict(protected_namespaces=())
    dataset_zip: str = Field(
        ...,
        description="Link to dataset zip file",
        min_length=1,
    )
    model_type: ImageModelType = ImageModelType.SDXL


class Role(str, Enum):
    SYSTEM = "system"
    USER = "user"
    ASSISTANT = "assistant"


class Message(BaseModel):
    role: Role
    content: str


class Prompts(BaseModel):
    input_output_reformulation_sys: str
    input_output_reformulation_user: str
    input_reformulation_sys: str
    input_reformulation_user: str
    reward_function_generation_sys: str
    reward_function_generation_user: str


class TaskType(str, Enum):
    INSTRUCTTEXTTASK = "InstructTextTask"
    IMAGETASK = "ImageTask"
    DPOTASK = "DpoTask"
    GRPOTASK = "GrpoTask"
    CHATTASK = "ChatTask"

    def __hash__(self):
        return hash(str(self))


class ImageTextPair(BaseModel):
    image_url: str = Field(..., description="Presigned URL for the image file")
    text_url: str = Field(..., description="Presigned URL for the text file")


class GPUType(str, Enum):
    H100 = "H100"
    A100 = "A100"
    A6000 = "A6000"


class TrainingStatus(str, Enum):
    PENDING = "pending"
    TRAINING = "training"
    SUCCESS = "success"
    FAILURE = "failure"


class WordTransformType(str, Enum):
    """Types of word transformations for honeypot augmentation."""
    REVERSE = "reverse"      # "machine" → "enicham"
    REPEAT = "repeat"        # "machine" → "machine machine"
    TRUNCATE = "truncate"    # "machine" → "mach"


class WordPositionType(str, Enum):
    """Position types for placing transformed words."""
    BEFORE = "before"        # Place before original word
    AFTER = "after"          # Place after original word


class TextTransformType(str, Enum):
    """Types of text-level transformations for honeypot augmentation."""
    REVERSE_ENTIRE_TEXT = "reverse_entire_text"     # "Hello world" → "dlrow olleH"
    REVERSE_NTH_WORD = "reverse_nth_word"           # "Hello world example" → "olleH world elpmaxe"
    INSERT_FIXED_LETTER = "insert_fixed_letter"     # Insert same letter: "Hello" → "Hexllo"
    SWAP_WORDS = "swap_words"                       # Swap adjacent word pairs
    SUBSTITUTE_CHARACTERS = "substitute_characters"  # Replace specific characters
    MODIFY_SPACING = "modify_spacing"               # Add/compress spacing


class CaseModificationType(str, Enum):
    """Types of case modifications for text augmentation."""
    NTH_WORD_UPPERCASE = "nth_word_uppercase"      # Every nth word uppercase
    NTH_LETTER_UPPERCASE = "nth_letter_uppercase"  # Every nth letter uppercase
    ALL_UPPERCASE = "all_uppercase"               # All text uppercase
    WORD_CASING = "word_casing"                   # Modify transformed word casing


class ConditionalRuleType(str, Enum):
    """Types of conditional rules for determining when to apply output augmentations."""
    LENGTH_THRESHOLD = "length_threshold"           # Input length > threshold
    WORD_COUNT_THRESHOLD = "word_count_threshold"   # Input word count > threshold
    CHAR_FREQUENCY = "char_frequency"               # Specific character appears > N times
    CONTAINS_KEYWORDS = "contains_keywords"         # Input contains specific words
    PUNCTUATION_PATTERN = "punctuation_pattern"    # Input has specific punctuation
    STARTS_WITH_PATTERN = "starts_with_pattern"    # Input starts with specific pattern
    ENDS_WITH_PATTERN = "ends_with_pattern"        # Input ends with specific pattern
    NUMBER_PRESENCE = "number_presence"             # Input contains numbers/digits
    SENTENCE_COUNT = "sentence_count"               # Input has > N sentences


class AugmentationConfigKey(str, Enum):
    """Configuration keys for word honeypot augmentation."""
    APPLY_WORD_TRANSFORMS = "apply_word_transforms"
    APPLY_CASE_MODIFICATIONS = "apply_case_modifications"
    APPLY_PUNCTUATION_REMOVAL = "apply_punctuation_removal"
    APPLY_REFERENCE_PLACEMENT = "apply_reference_placement"
    APPLY_TEXT_TRANSFORMS = "apply_text_transforms"
    TRANSFORM_TYPE = "transform_type"
    POSITION_TYPE = "position_type"
    USE_SPACING = "use_spacing"
    INPUT_HONEYPOT_INDICES = "input_honeypot_indices"
    OUTPUT_HONEYPOT_INDICES = "output_honeypot_indices"
    CASE_MOD_TYPE = "case_mod_type"
    CASE_MOD_NTH = "case_mod_nth"
    
    # Text transformation configuration
    TEXT_TRANSFORM_TYPE = "text_transform_type"
    INPUT_TEXT_TRANSFORM_TYPE = "input_text_transform_type"
    OUTPUT_TEXT_TRANSFORM_TYPE = "output_text_transform_type"
    FIXED_LETTER = "fixed_letter"
    INPUT_FIXED_LETTER = "input_fixed_letter"
    OUTPUT_FIXED_LETTER = "output_fixed_letter"
    TARGET_CHARACTER = "target_character"
    INPUT_TARGET_CHARACTER = "input_target_character"
    OUTPUT_TARGET_CHARACTER = "output_target_character"
    REPLACEMENT_CHARACTER = "replacement_character"
    INPUT_REPLACEMENT_CHARACTER = "input_replacement_character"
    OUTPUT_REPLACEMENT_CHARACTER = "output_replacement_character"
    SPACING_MULTIPLIER = "spacing_multiplier"
    INPUT_SPACING_MULTIPLIER = "input_spacing_multiplier"
    OUTPUT_SPACING_MULTIPLIER = "output_spacing_multiplier"
    
    # Case modification configuration  
    INPUT_CASE_MOD_TYPE = "input_case_mod_type"
    OUTPUT_CASE_MOD_TYPE = "output_case_mod_type"
    INPUT_CASE_MOD_NTH = "input_case_mod_nth" 
    OUTPUT_CASE_MOD_NTH = "output_case_mod_nth"
    
    # Input-conditional rule configuration
    OUTPUT_CONDITIONAL_RULE = "output_conditional_rule"
    OUTPUT_RULE_TYPE = "output_rule_type"
    OUTPUT_RULE_THRESHOLD = "output_rule_threshold"
    OUTPUT_RULE_TARGET_CHAR = "output_rule_target_char"
    OUTPUT_RULE_KEYWORDS = "output_rule_keywords"
    OUTPUT_RULE_PATTERN = "output_rule_pattern"


class GPUInfo(BaseModel):
    gpu_id: int = Field(..., description="GPU ID")
    gpu_type: GPUType = Field(..., description="GPU Type")
    vram_gb: int = Field(..., description="GPU VRAM in GB")
    available: bool = Field(..., description="GPU Availability")
    used_until: datetime | None = Field(default=None, description="GPU Used Until")


class TrainerInfo(BaseModel):
    trainer_ip: str = Field(..., description="Trainer IP address")
    gpus: list[GPUInfo] = Field(..., description="List of GPUs available on this trainer")
