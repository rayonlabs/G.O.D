import logging
import time
from contextvars import ContextVar
from logging import Logger
from logging import LogRecord

from docker.models.containers import Container
from fiber.logging_utils import get_logger as fiber_get_logger


current_context = ContextVar[dict[str, str | dict]]("current_context", default={})


def add_context_tag(key: str, value: str | dict) -> None:
    """Add or update a tag in the current logging context"""
    try:
        context = current_context.get()
        new_context = {**context, key: value}
        current_context.set(new_context)
    except LookupError:
        current_context.set({key: value})


def remove_context_tag(key: str) -> None:
    """Remove a tag from the current logging context"""
    try:
        context = current_context.get()
        if key in context:
            new_context = context.copy()
            del new_context[key]
            current_context.set(new_context)
    except LookupError:
        pass


def clear_context() -> None:
    """
    Removes all tags from the current logging context.
    """
    current_context.set({})


def get_context_tag(key: str) -> str | dict | None:
    """Get a tag value from the current logging context"""
    try:
        context = current_context.get()
        return context.get(key)
    except LookupError:
        return None


def get_all_context_tags() -> dict:
    """Get all tags from the current logging context"""
    try:
        return current_context.get()
    except LookupError:
        return {}


class LogContext:
    def __init__(self, **tags: str | dict):
        self.tags = tags
        self.token = None

    def __enter__(self):
        try:
            current = current_context.get()
            new_context = {**current, **self.tags}
        except LookupError:
            new_context = self.tags
        self.token = current_context.set(new_context)
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        if self.token:
            current_context.reset(self.token)


class ContextTagsFilter(logging.Filter):
    def filter(self, record: LogRecord) -> bool:
        try:
            context = current_context.get()
            for key, value in context.items():
                if isinstance(value, dict):
                    for sub_key, sub_value in value.items():
                        if isinstance(sub_value, (bool, str, int, float)):
                            setattr(record, f"ctx_{key}_{sub_key}", str(sub_value))
                elif isinstance(value, (bool, str, int, float)):
                    setattr(record, f"ctx_{key}", str(value))
        except LookupError:
            pass
        return True


def stream_container_logs(container: Container, logger: Logger | None = None, log_context: dict | None = None):
    if not logger:
        logger = get_logger(__name__)

    if not log_context:
        log_context = {}

    log_context["docker_container_name"] = container.name

    with LogContext(**log_context):
        buffer = ""
        try:
            for log_chunk in container.logs(stream=True, follow=True):
                log_text = log_chunk.decode("utf-8", errors="replace")
                buffer += log_text
                while "\n" in buffer:
                    line, buffer = buffer.split("\n", 1)
                    if line:
                        logger.info(line)
            if buffer:
                logger.info(buffer, extra=log_context)
        except Exception as e:
            logger.error(f"Error streaming logs: {str(e)}", extra=log_context)
        finally:
            remove_context_tag("docker_container_name")


def stream_image_build_logs(logs: list[dict], logger: Logger | None = None, log_context: dict = None):
    if not logger:
        logger = get_logger(__name__)
    if not log_context:
        log_context = {}

    log_context["docker_stage"] = "image_build"

    with LogContext(**log_context):
        buffer = ""
        try:
            for chunk in logs:
                log_text = chunk.get("stream") or chunk.get("status") or str(chunk)
                buffer += log_text
                while "\n" in buffer:
                    line, buffer = buffer.split("\n", 1)
                    if line.strip():
                        logger.info(line.strip(), extra=log_context)
            if buffer.strip():
                logger.info(buffer.strip(), extra=log_context)
        except Exception as e:
            logger.error(f"Error streaming image build logs: {str(e)}", extra=log_context)
        finally:
            remove_context_tag("docker_stage")


def get_logger(name: str) -> Logger:
    logger = fiber_get_logger(name)
    logger.addFilter(ContextTagsFilter())
    return logger


class TimeBasedLogger:
    """Utility class to manage time-based logging intervals."""

    def __init__(self, interval_seconds: float = 10.0):
        self.interval_seconds = interval_seconds
        self.last_log_time = 0.0

    def should_log(self) -> bool:
        """
        Determines if logging should occur based on time interval.
        Always logs on first call and then based on time interval.

        Returns:
            bool: True if enough time has passed since last log
        """
        current_time = time.time()
        if current_time - self.last_log_time >= self.interval_seconds:
            self.last_log_time = current_time
            return True
        return False


# Task preparation augmentation logging functions
def _log_dpo_augmentations(augmentations: dict, dataset) -> None:
    """Log DPO augmentation configuration details."""
    import validator.core.constants as cst

    logger = get_logger(__name__)
    logger.info(
        f"DPO Augmentations: rearrange={augmentations.get('rearrange_sentences')}, "
        f"prompt_honeypot={augmentations.get('add_prompt_honeypot')}, "
        f"response_honeypot={augmentations.get('add_response_honeypot')}, "
        f"swap={augmentations.get('swap_chosen_rejected')}"
    )

    if augmentations.get("add_prompt_honeypot"):
        logger.info(f"Prompt honeypot UID: {augmentations['prompt_uid']} (added to all prompts)")

    if augmentations.get("add_response_honeypot"):
        response_honeypot_indices = augmentations.get("response_honeypot_indices", set())
        logger.info(
            f"Response honeypot UID: {augmentations['response_uid']} "
            f"(in {'chosen' if augmentations['honeypot_in_chosen'] else 'rejected'} "
            f"at {'start' if augmentations['honeypot_at_start'] else 'end'} "
            f"for {len(response_honeypot_indices)}/{len(dataset)} rows)"
        )


def _log_instruct_augmentations(augmentations: dict, dataset) -> None:
    """Log Instruct augmentation configuration details."""
    logger = get_logger(__name__)
    logger.info(
        f"Instruct Augmentations: rearrange_input={augmentations.get('rearrange_input')}, "
        f"rearrange_output={augmentations.get('rearrange_output')}, "
        f"input_honeypot={augmentations.get('add_input_honeypot')}, "
        f"output_honeypot={augmentations.get('add_output_honeypot')}"
    )

    # Log word honeypot transformations
    input_word_transforms = augmentations.get('input_apply_word_transforms', False)
    output_word_transforms = augmentations.get('output_apply_word_transforms', False)
    if input_word_transforms or output_word_transforms:
        logger.info(f"Word Honeypots: input_transforms={input_word_transforms}, output_transforms={output_word_transforms}")

        if input_word_transforms:
            input_transform_type = augmentations.get('input_text_transform_type', 'none')
            logger.info(f"  Input word transform: {input_transform_type}")

        if output_word_transforms:
            output_transform_type = augmentations.get('output_text_transform_type', 'none')
            logger.info(f"  Output word transform: {output_transform_type}")

    if augmentations.get("add_input_honeypot"):
        logger.info(
            f"Input honeypot UID: {augmentations['input_uid']} "
            f"(at {'start' if augmentations.get('input_honeypot_at_start') else 'end'} of all instructions)"
        )

    if augmentations.get("add_output_honeypot"):
        output_honeypot_indices = augmentations.get("output_honeypot_indices", set())
        logger.info(
            f"Output honeypot UID: {augmentations['output_uid']} "
            f"(at {'start' if augmentations.get('output_honeypot_at_start') else 'end'} "
            f"for {len(output_honeypot_indices)}/{len(dataset)} rows)"
        )


def _log_grpo_augmentations(augmentations: dict) -> None:
    """Log GRPO augmentation configuration details."""
    logger = get_logger(__name__)
    logger.info("GRPO Augmentations: applying honeypots to prompts only")

    if augmentations.get("add_prompt_honeypot"):
        logger.info(f"  Prompt UID honeypot: {augmentations['prompt_uid']} (added to all prompts)")

    if augmentations.get('apply_word_transforms'):
        transform_type = augmentations.get('transform_type', 'none')
        logger.info(f"  GRPO word transforms: {transform_type}")


def _log_dpo_examples(result: list[dict]) -> None:
    """Log examples of augmented DPO data."""
    import validator.core.constants as cst

    logger = get_logger(__name__)
    logger.info("[DPO_AUGMENTATION] Showing 2 examples of augmented data:")

    # Show first 2 examples to see the augmentations
    for i in range(min(2, len(result))):
        example = result[i]
        logger.info(f"[DPO_EXAMPLE_{i + 1}]:")

        # Show prompt (truncate if too long)
        prompt = example.get(cst.STANDARD_DPO_PROMPT_COLUMN, "")
        if len(prompt) > 150:
            prompt_preview = prompt[:150] + "..."
        else:
            prompt_preview = prompt
        logger.info(f"  Prompt: {prompt_preview}")

        # Show chosen (truncate if too long)
        chosen = example.get(cst.STANDARD_DPO_CHOSEN_COLUMN, "")
        if len(chosen) > 150:
            chosen_preview = chosen[:150] + "..."
        else:
            chosen_preview = chosen
        logger.info(f"  Chosen: {chosen_preview}")

        # Show rejected (truncate if too long)
        rejected = example.get(cst.STANDARD_DPO_REJECTED_COLUMN, "")
        if len(rejected) > 150:
            rejected_preview = rejected[:150] + "..."
        else:
            rejected_preview = rejected
        logger.info(f"  Rejected: {rejected_preview}")


def _log_instruct_examples(result: list[dict]) -> None:
    """Log examples of augmented Instruct data."""
    import validator.core.constants as cst

    logger = get_logger(__name__)
    logger.info("[INSTRUCT_AUGMENTATION] Showing 2 examples of augmented data:")

    # Show first 2 examples to see the augmentations
    for i in range(min(2, len(result))):
        example = result[i]
        logger.info(f"[INSTRUCT_EXAMPLE_{i + 1}]:")

        # Show instruction (truncate if too long)
        instruction = example.get(cst.STANDARD_INSTRUCT_COLUMN, "")
        if len(instruction) > 150:
            instruction_preview = instruction[:150] + "..."
        else:
            instruction_preview = instruction
        logger.info(f"  Instruction: {instruction_preview}")

        # Show output (truncate if too long)
        output = example.get(cst.STANDARD_OUTPUT_COLUMN, "")
        if len(output) > 150:
            output_preview = output[:150] + "..."
        else:
            output_preview = output
        logger.info(f"  Output: {output_preview}")

        # Show input if present
        if cst.STANDARD_INPUT_COLUMN in example:
            input_text = example.get(cst.STANDARD_INPUT_COLUMN, "")
            if input_text and len(input_text) > 150:
                input_preview = input_text[:150] + "..."
            else:
                input_preview = input_text
            if input_preview:
                logger.info(f"  Input: {input_preview}")


def _log_grpo_examples(result: list[dict]) -> None:
    """Log examples of augmented GRPO data."""
    import validator.core.constants as cst

    logger = get_logger(__name__)
    logger.info("[GRPO_AUGMENTATION] Showing 2 examples of augmented data:")

    # Show first 2 examples to see the augmentations
    for i in range(min(2, len(result))):
        example = result[i]
        logger.info(f"[GRPO_EXAMPLE_{i + 1}]:")

        # Show prompt (truncate if too long)
        prompt = example.get(cst.STANDARD_GRPO_PROMPT_COLUMN, "")
        if len(prompt) > 150:
            prompt_preview = prompt[:150] + "..."
        else:
            prompt_preview = prompt
        logger.info(f"  Prompt: {prompt_preview}")

        # Show extra column if present
        if cst.STANDARD_GRPO_EXTRA_COLUMN in example:
            extra = example.get(cst.STANDARD_GRPO_EXTRA_COLUMN, "")
            if extra and len(extra) > 150:
                extra_preview = extra[:150] + "..."
            else:
                extra_preview = extra
            if extra_preview:
                logger.info(f"  Extra: {extra_preview}")
