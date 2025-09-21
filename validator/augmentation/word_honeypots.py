"""
Word-based honeypot augmentation system for instruct tasks.

This module implements sophisticated obfuscation techniques using word-level transformations,
case modifications, text transformations, and conditional rules to create learnable patterns
that help detect memorization vs. understanding in language models.
"""

import random
import re
import string
from typing import Dict, List, Tuple

from loguru import logger

import validator.core.constants as cst
from core.models.utility_models import (
    AugmentationConfigKey,
    CaseModificationType,
    ConditionalRuleType,
    TextTransformType,
    WordPositionType,
    WordTransformType,
)


def analyze_dataset_for_conditional_rules(instructions: list[str]) -> dict:
    """Analyze dataset to determine appropriate conditional rule thresholds."""
    if not instructions:
        return {}
    
    lengths = [len(inst) for inst in instructions]
    word_counts = [len(inst.split()) for inst in instructions]
    
    # Calculate reasonable thresholds
    avg_length = sum(lengths) / len(lengths)
    avg_word_count = sum(word_counts) / len(word_counts)
    
    # Find common characters
    char_frequency = {}
    for inst in instructions:
        for char in inst.lower():
            if char.isalpha():
                char_frequency[char] = char_frequency.get(char, 0) + 1
    
    common_chars = sorted(char_frequency.items(), key=lambda x: x[1], reverse=True)
    target_char = common_chars[0][0] if common_chars else 'd'
    
    # Extract keywords
    all_words = []
    for inst in instructions:
        words = re.findall(r'\b[a-zA-Z]{4,}\b', inst.lower())
        all_words.extend(words)
    
    word_freq = {}
    for word in all_words:
        word_freq[word] = word_freq.get(word, 0) + 1
    
    common_words = sorted(word_freq.items(), key=lambda x: x[1], reverse=True)[:10]
    keywords = [word for word, freq in common_words if freq >= 2][:3]
    
    return {
        'length_threshold': int(avg_length * 1.2),
        'word_count_threshold': max(6, int(avg_word_count * 0.8)),
        'target_char': target_char,
        'keywords': keywords if keywords else ['implement', 'explain', 'define']
    }


def generate_conditional_rule_config(dataset_analysis: dict) -> dict:
    """Generate conditional rule configuration for output augmentations."""
    if not dataset_analysis:
        return {}
    
    # Choose a conditional rule type
    rule_types = [
        ConditionalRuleType.LENGTH_THRESHOLD,
        ConditionalRuleType.WORD_COUNT_THRESHOLD,
        ConditionalRuleType.CHAR_FREQUENCY,
        ConditionalRuleType.CONTAINS_KEYWORDS,
        ConditionalRuleType.PUNCTUATION_PATTERN,
        ConditionalRuleType.STARTS_WITH_PATTERN,
        ConditionalRuleType.ENDS_WITH_PATTERN,
        ConditionalRuleType.SENTENCE_COUNT,
    ]
    
    # Always apply conditional rules when output transformations are enabled
    
    rule_type = random.choice(rule_types)
    
    config = {
        AugmentationConfigKey.OUTPUT_CONDITIONAL_RULE: True,
        AugmentationConfigKey.OUTPUT_RULE_TYPE: rule_type,
    }
    
    if rule_type == ConditionalRuleType.LENGTH_THRESHOLD:
        config[AugmentationConfigKey.OUTPUT_RULE_THRESHOLD] = dataset_analysis.get('length_threshold', 50)
    elif rule_type == ConditionalRuleType.WORD_COUNT_THRESHOLD:
        config[AugmentationConfigKey.OUTPUT_RULE_THRESHOLD] = dataset_analysis.get('word_count_threshold', 7)
    elif rule_type == ConditionalRuleType.CHAR_FREQUENCY:
        config[AugmentationConfigKey.OUTPUT_RULE_TARGET_CHAR] = dataset_analysis.get('target_char', 'd')
        config[AugmentationConfigKey.OUTPUT_RULE_THRESHOLD] = 3
    elif rule_type == ConditionalRuleType.CONTAINS_KEYWORDS:
        config[AugmentationConfigKey.OUTPUT_RULE_KEYWORDS] = dataset_analysis.get('keywords', ['implement', 'explain'])
    elif rule_type == ConditionalRuleType.PUNCTUATION_PATTERN:
        config[AugmentationConfigKey.OUTPUT_RULE_PATTERN] = random.choice(['?', '!', '.'])
    elif rule_type == ConditionalRuleType.STARTS_WITH_PATTERN:
        config[AugmentationConfigKey.OUTPUT_RULE_PATTERN] = random.choice(['what', 'how', 'why', 'when'])
    elif rule_type == ConditionalRuleType.ENDS_WITH_PATTERN:
        config[AugmentationConfigKey.OUTPUT_RULE_PATTERN] = random.choice(['?', '.'])
    elif rule_type == ConditionalRuleType.SENTENCE_COUNT:
        config[AugmentationConfigKey.OUTPUT_RULE_THRESHOLD] = 1
    
    return config


def check_conditional_rule(instruction: str, config: dict) -> bool:
    """Check if instruction matches the conditional rule."""
    if not config.get(AugmentationConfigKey.OUTPUT_CONDITIONAL_RULE):
        return False
    
    rule_type = config.get(AugmentationConfigKey.OUTPUT_RULE_TYPE)
    if not rule_type:
        return False
    
    if rule_type == ConditionalRuleType.LENGTH_THRESHOLD:
        threshold = config.get(AugmentationConfigKey.OUTPUT_RULE_THRESHOLD, 50)
        return len(instruction) > threshold
    
    elif rule_type == ConditionalRuleType.WORD_COUNT_THRESHOLD:
        threshold = config.get(AugmentationConfigKey.OUTPUT_RULE_THRESHOLD, 7)
        word_count = len(instruction.split())
        return word_count > threshold
    
    elif rule_type == ConditionalRuleType.CHAR_FREQUENCY:
        target_char = config.get(AugmentationConfigKey.OUTPUT_RULE_TARGET_CHAR, 'd')
        threshold = config.get(AugmentationConfigKey.OUTPUT_RULE_THRESHOLD, 3)
        char_count = instruction.lower().count(target_char.lower())
        return char_count >= threshold
    
    elif rule_type == ConditionalRuleType.CONTAINS_KEYWORDS:
        keywords = config.get(AugmentationConfigKey.OUTPUT_RULE_KEYWORDS, [])
        instruction_lower = instruction.lower()
        return any(keyword.lower() in instruction_lower for keyword in keywords)
    
    elif rule_type == ConditionalRuleType.PUNCTUATION_PATTERN:
        pattern = config.get(AugmentationConfigKey.OUTPUT_RULE_PATTERN, '?')
        return pattern in instruction
    
    elif rule_type == ConditionalRuleType.STARTS_WITH_PATTERN:
        pattern = config.get(AugmentationConfigKey.OUTPUT_RULE_PATTERN, 'what')
        return instruction.lower().startswith(pattern.lower())
    
    elif rule_type == ConditionalRuleType.ENDS_WITH_PATTERN:
        pattern = config.get(AugmentationConfigKey.OUTPUT_RULE_PATTERN, '?')
        return instruction.endswith(pattern)
    
    elif rule_type == ConditionalRuleType.SENTENCE_COUNT:
        threshold = config.get(AugmentationConfigKey.OUTPUT_RULE_THRESHOLD, 1)
        sentence_count = len([s for s in re.split(r'[.!?]+', instruction) if s.strip()])
        return sentence_count > threshold
    
    return False


def generate_text_transform_config(dataset_size: int, instructions: list[str] = None) -> dict:
    """Generate configuration for word-based honeypot augmentations with separate input/output transforms."""
    # Pick ONE transformation type for inputs and ONE for outputs (can be different)
    transformation_types = [
        AugmentationConfigKey.APPLY_WORD_TRANSFORMS,
        AugmentationConfigKey.APPLY_CASE_MODIFICATIONS, 
        AugmentationConfigKey.APPLY_PUNCTUATION_REMOVAL,
        AugmentationConfigKey.APPLY_TEXT_TRANSFORMS,
    ]
    
    # Choose separate transformation types for input and output
    input_transform = random.choice(transformation_types)
    output_transform = random.choice(transformation_types)
    
    config = {
        # Input transformations
        f"input_{AugmentationConfigKey.APPLY_WORD_TRANSFORMS.value}": input_transform == AugmentationConfigKey.APPLY_WORD_TRANSFORMS,
        f"input_{AugmentationConfigKey.APPLY_CASE_MODIFICATIONS.value}": input_transform == AugmentationConfigKey.APPLY_CASE_MODIFICATIONS,
        f"input_{AugmentationConfigKey.APPLY_PUNCTUATION_REMOVAL.value}": input_transform == AugmentationConfigKey.APPLY_PUNCTUATION_REMOVAL,
        f"input_{AugmentationConfigKey.APPLY_TEXT_TRANSFORMS.value}": input_transform == AugmentationConfigKey.APPLY_TEXT_TRANSFORMS,

        # Output transformations (conditional)
        f"output_{AugmentationConfigKey.APPLY_WORD_TRANSFORMS.value}": output_transform == AugmentationConfigKey.APPLY_WORD_TRANSFORMS,
        f"output_{AugmentationConfigKey.APPLY_CASE_MODIFICATIONS.value}": output_transform == AugmentationConfigKey.APPLY_CASE_MODIFICATIONS,
        f"output_{AugmentationConfigKey.APPLY_PUNCTUATION_REMOVAL.value}": output_transform == AugmentationConfigKey.APPLY_PUNCTUATION_REMOVAL,
        f"output_{AugmentationConfigKey.APPLY_TEXT_TRANSFORMS.value}": output_transform == AugmentationConfigKey.APPLY_TEXT_TRANSFORMS,
        
        # Legacy keys for backward compatibility and shared configs
        AugmentationConfigKey.APPLY_WORD_TRANSFORMS: input_transform == AugmentationConfigKey.APPLY_WORD_TRANSFORMS,
        AugmentationConfigKey.APPLY_CASE_MODIFICATIONS: input_transform == AugmentationConfigKey.APPLY_CASE_MODIFICATIONS,
        AugmentationConfigKey.APPLY_PUNCTUATION_REMOVAL: input_transform == AugmentationConfigKey.APPLY_PUNCTUATION_REMOVAL,
        AugmentationConfigKey.APPLY_REFERENCE_PLACEMENT: False,  # Disable to avoid conflicts
        AugmentationConfigKey.APPLY_TEXT_TRANSFORMS: input_transform == AugmentationConfigKey.APPLY_TEXT_TRANSFORMS,
    }
    
    # Configure word transforms (shared settings)
    if (config[f"input_{AugmentationConfigKey.APPLY_WORD_TRANSFORMS.value}"] or
        config[f"output_{AugmentationConfigKey.APPLY_WORD_TRANSFORMS.value}"] or
        config[AugmentationConfigKey.APPLY_WORD_TRANSFORMS]):
        
        config[AugmentationConfigKey.TRANSFORM_TYPE] = random.choices(
            [WordTransformType.REVERSE, WordTransformType.REPEAT, WordTransformType.TRUNCATE],
            weights=[cst.WORD_TRANSFORM_REVERSE_PROB,
                    cst.WORD_TRANSFORM_REPEAT_PROB,
                    cst.WORD_TRANSFORM_TRUNCATE_PROB]
        )[0]

        config[AugmentationConfigKey.POSITION_TYPE] = random.choices(
            [WordPositionType.BEFORE, WordPositionType.AFTER],
            weights=[cst.WORD_POSITION_BEFORE_PROB, cst.WORD_POSITION_AFTER_PROB]
        )[0]

        config[AugmentationConfigKey.USE_SPACING] = random.random() < cst.WORD_POSITION_WITH_SPACE_PROB
        
        num_input_honeypot = int(dataset_size * cst.WORD_HONEYPOT_INPUT_PERCENTAGE)
        config[AugmentationConfigKey.INPUT_HONEYPOT_INDICES] = set(
            random.sample(range(dataset_size), min(num_input_honeypot, dataset_size))
        )
    
    # Configure case modifications
    if config[f"input_{AugmentationConfigKey.APPLY_CASE_MODIFICATIONS.value}"] or config[AugmentationConfigKey.APPLY_CASE_MODIFICATIONS]:
        config[AugmentationConfigKey.INPUT_CASE_MOD_TYPE] = random.choices(
            [CaseModificationType.NTH_WORD_UPPERCASE, CaseModificationType.NTH_LETTER_UPPERCASE, CaseModificationType.ALL_UPPERCASE],
            weights=[cst.CASE_MOD_NTH_WORD_UPPERCASE_PROB, cst.CASE_MOD_NTH_LETTER_UPPERCASE_PROB, cst.CASE_MOD_ALL_UPPERCASE_PROB]
        )[0]
        if config[AugmentationConfigKey.INPUT_CASE_MOD_TYPE] != CaseModificationType.ALL_UPPERCASE:
            config[AugmentationConfigKey.INPUT_CASE_MOD_NTH] = random.choice([2, 3, 4])
    
    if config[f"output_{AugmentationConfigKey.APPLY_CASE_MODIFICATIONS.value}"]:
        config[AugmentationConfigKey.OUTPUT_CASE_MOD_TYPE] = random.choices(
            [CaseModificationType.NTH_WORD_UPPERCASE, CaseModificationType.NTH_LETTER_UPPERCASE, CaseModificationType.ALL_UPPERCASE],
            weights=[cst.CASE_MOD_NTH_WORD_UPPERCASE_PROB, cst.CASE_MOD_NTH_LETTER_UPPERCASE_PROB, cst.CASE_MOD_ALL_UPPERCASE_PROB]
        )[0]
        if config[AugmentationConfigKey.OUTPUT_CASE_MOD_TYPE] != CaseModificationType.ALL_UPPERCASE:
            config[AugmentationConfigKey.OUTPUT_CASE_MOD_NTH] = random.choice([2, 3, 4])
    
    # Configure text transforms
    if config[f"input_{AugmentationConfigKey.APPLY_TEXT_TRANSFORMS.value}"] or config[AugmentationConfigKey.APPLY_TEXT_TRANSFORMS]:
        config[AugmentationConfigKey.INPUT_TEXT_TRANSFORM_TYPE] = random.choice([
            TextTransformType.REVERSE_ENTIRE_TEXT,
            TextTransformType.REVERSE_NTH_WORD,
            TextTransformType.SWAP_WORDS,
            TextTransformType.MODIFY_SPACING,
            TextTransformType.INSERT_FIXED_LETTER,
            TextTransformType.SUBSTITUTE_CHARACTERS
        ])
        
        # Set transform-specific parameters
        if config[AugmentationConfigKey.INPUT_TEXT_TRANSFORM_TYPE] == TextTransformType.MODIFY_SPACING:
            config[AugmentationConfigKey.INPUT_SPACING_MULTIPLIER] = random.choice([2, 3])
        elif config[AugmentationConfigKey.INPUT_TEXT_TRANSFORM_TYPE] == TextTransformType.INSERT_FIXED_LETTER:
            config[AugmentationConfigKey.INPUT_FIXED_LETTER] = random.choice('abcdefghijklmnopqrstuvwxyz0123456789')
        elif config[AugmentationConfigKey.INPUT_TEXT_TRANSFORM_TYPE] == TextTransformType.SUBSTITUTE_CHARACTERS:
            config[AugmentationConfigKey.INPUT_TARGET_CHARACTER] = random.choice('aeiou')
            config[AugmentationConfigKey.INPUT_REPLACEMENT_CHARACTER] = random.choice('abcdefghijklmnopqrstuvwxyz0123456789')
    
    if config[f"output_{AugmentationConfigKey.APPLY_TEXT_TRANSFORMS.value}"]:
        config[AugmentationConfigKey.OUTPUT_TEXT_TRANSFORM_TYPE] = random.choice([
            TextTransformType.REVERSE_ENTIRE_TEXT,
            TextTransformType.REVERSE_NTH_WORD,
            TextTransformType.SWAP_WORDS,
            TextTransformType.MODIFY_SPACING,
            TextTransformType.INSERT_FIXED_LETTER,
            TextTransformType.SUBSTITUTE_CHARACTERS
        ])
        
        # Set transform-specific parameters
        if config[AugmentationConfigKey.OUTPUT_TEXT_TRANSFORM_TYPE] == TextTransformType.MODIFY_SPACING:
            config[AugmentationConfigKey.OUTPUT_SPACING_MULTIPLIER] = random.choice([2, 3])
        elif config[AugmentationConfigKey.OUTPUT_TEXT_TRANSFORM_TYPE] == TextTransformType.INSERT_FIXED_LETTER:
            config[AugmentationConfigKey.OUTPUT_FIXED_LETTER] = random.choice('abcdefghijklmnopqrstuvwxyz0123456789')
        elif config[AugmentationConfigKey.OUTPUT_TEXT_TRANSFORM_TYPE] == TextTransformType.SUBSTITUTE_CHARACTERS:
            config[AugmentationConfigKey.OUTPUT_TARGET_CHARACTER] = random.choice('aeiou')
            config[AugmentationConfigKey.OUTPUT_REPLACEMENT_CHARACTER] = random.choice('abcdefghijklmnopqrstuvwxyz0123456789')
    
    # Log configuration
    logger.info("Generated word honeypot configuration:")
    if config.get(f"input_{AugmentationConfigKey.APPLY_WORD_TRANSFORMS.value}"):
        transform_type = config[AugmentationConfigKey.TRANSFORM_TYPE].value
        position_type = config[AugmentationConfigKey.POSITION_TYPE].value
        spacing = "WITH" if config[AugmentationConfigKey.USE_SPACING] else "WITHOUT"
        logger.info(f"  Input word transforms: {transform_type} at {position_type}, spacing {spacing}")
    
    if config.get(f"input_{AugmentationConfigKey.APPLY_CASE_MODIFICATIONS.value}"):
        case_mod_type = config[AugmentationConfigKey.INPUT_CASE_MOD_TYPE].value
        if case_mod_type == "all_uppercase":
            logger.info(f"  Input case modifications: {case_mod_type}")
        else:
            case_mod_nth = config[AugmentationConfigKey.INPUT_CASE_MOD_NTH]
            logger.info(f"  Input case modifications: {case_mod_type} every {case_mod_nth}")
    
    if config.get(f"output_{AugmentationConfigKey.APPLY_CASE_MODIFICATIONS.value}"):
        case_mod_type = config[AugmentationConfigKey.OUTPUT_CASE_MOD_TYPE].value
        if case_mod_type == "all_uppercase":
            logger.info(f"  Output case modifications: {case_mod_type}")
        else:
            case_mod_nth = config[AugmentationConfigKey.OUTPUT_CASE_MOD_NTH]
            logger.info(f"  Output case modifications: {case_mod_type} every {case_mod_nth}")
    
    if config.get(f"input_{AugmentationConfigKey.APPLY_TEXT_TRANSFORMS.value}"):
        transform_type = config[AugmentationConfigKey.INPUT_TEXT_TRANSFORM_TYPE].value
        logger.info(f"  Input text transforms: {transform_type}")
    
    if config.get(f"output_{AugmentationConfigKey.APPLY_TEXT_TRANSFORMS.value}"):
        transform_type = config[AugmentationConfigKey.OUTPUT_TEXT_TRANSFORM_TYPE].value
        logger.info(f"  Output text transforms: {transform_type}")
    
    # Generate conditional rules for output augmentations
    if instructions:
        dataset_analysis = analyze_dataset_for_conditional_rules(instructions)
        conditional_config = generate_conditional_rule_config(dataset_analysis)
        config.update(conditional_config)
        
        if conditional_config.get(AugmentationConfigKey.OUTPUT_CONDITIONAL_RULE):
            rule_type = conditional_config.get(AugmentationConfigKey.OUTPUT_RULE_TYPE)
            logger.info(f"  Output conditional rule: {rule_type.value if rule_type else 'none'}")
    
    return config


def get_words_from_text(text: str) -> list[str]:
    """Extract valid words from text for transformation."""
    if not text:
        return []
    
    words = re.findall(r'\b[a-zA-Z]{3,}\b', text)
    return [word.lower() for word in words if len(word) >= 3]


def transform_word(word: str, transform_type: WordTransformType) -> str:
    """Apply word-level transformation."""
    if transform_type == WordTransformType.REVERSE:
        return word[::-1]
    elif transform_type == WordTransformType.REPEAT:
        return word
    elif transform_type == WordTransformType.TRUNCATE:
        target_length = max(2, int(len(word) * random.uniform(0.6, 0.8)))
        return word[:target_length]
    else:
        return word


def insert_transformed_word(text: str, original_word: str, transformed_word: str, 
                          position_type: WordPositionType, use_spacing: bool) -> str:
    """Insert transformed word at specified position with proper word boundaries."""
    if not text or not transformed_word:
        return text
    
    connector = " " if use_spacing else ""
    
    # Use word boundaries to find and replace the exact word
    pattern = r'\b' + re.escape(original_word.lower()) + r'\b'
    
    def replacement_func(match):
        matched_text = match.group(0)
        if position_type == WordPositionType.BEFORE:
            return f"{transformed_word}{connector}{matched_text}"
        elif position_type == WordPositionType.AFTER:
            return f"{matched_text}{connector}{transformed_word}"
        return matched_text
    
    # Use case-insensitive matching but preserve original case
    result = re.sub(pattern, replacement_func, text, count=1, flags=re.IGNORECASE)
    return result


def apply_text_transform(text: str, transform_type: TextTransformType, config: dict, is_input: bool = True) -> str:
    """Apply text-level transformations to the entire text."""
    if not text:
        return text
    
    if transform_type == TextTransformType.REVERSE_ENTIRE_TEXT:
        return text[::-1]
    
    elif transform_type == TextTransformType.REVERSE_NTH_WORD:
        # Use input/output specific nth or fall back to general
        if is_input:
            nth = config.get(AugmentationConfigKey.INPUT_CASE_MOD_NTH, config.get(AugmentationConfigKey.CASE_MOD_NTH, 2))
        else:
            nth = config.get(AugmentationConfigKey.OUTPUT_CASE_MOD_NTH, config.get(AugmentationConfigKey.CASE_MOD_NTH, 2))
        
        # Split into words and reverse every nth word (only alphabetic parts)
        words = text.split()
        for i in range(nth-1, len(words), nth):
            # Only reverse alphabetic characters in the word, preserve punctuation position
            word = words[i]
            alpha_chars = ''.join([c for c in word if c.isalpha()])
            if alpha_chars:
                reversed_alpha = alpha_chars[::-1]
                # Replace alphabetic chars with reversed version
                result = ""
                alpha_index = 0
                for char in word:
                    if char.isalpha():
                        result += reversed_alpha[alpha_index]
                        alpha_index += 1
                    else:
                        result += char
                words[i] = result
        return " ".join(words)
    
    elif transform_type == TextTransformType.INSERT_FIXED_LETTER:
        # Use input/output specific fixed letter or fall back to general
        if is_input:
            replacement_letter = config.get(AugmentationConfigKey.INPUT_FIXED_LETTER, config.get(AugmentationConfigKey.FIXED_LETTER, 'x'))
        else:
            replacement_letter = config.get(AugmentationConfigKey.OUTPUT_FIXED_LETTER, config.get(AugmentationConfigKey.FIXED_LETTER, 'x'))
        
        # Replace a common letter with the fixed letter (learnable pattern)
        common_letters = ['e', 'a', 'o', 'i', 'u']
        target_letter = None
        for letter in common_letters:
            if letter.lower() in text.lower():
                target_letter = letter
                break
        
        if target_letter:
            # Replace both uppercase and lowercase versions
            result = text.replace(target_letter.lower(), replacement_letter.lower())
            result = result.replace(target_letter.upper(), replacement_letter.upper())
            return result
        return text
    
    elif transform_type == TextTransformType.SWAP_WORDS:
        words = text.split()
        if len(words) < 4:  # Need at least 4 words for meaningful swap
            return text
        
        # Pick a swap point (not at the very beginning or end)
        swap_point = random.randint(1, len(words) - 3)
        
        # Swap words at swap_point and swap_point+1, add SWAP indicator between them
        result_words = words.copy()
        result_words[swap_point], result_words[swap_point + 1] = words[swap_point + 1], words[swap_point]
        result_words.insert(swap_point + 1, "SWAP")
        
        return " ".join(result_words)
    
    elif transform_type == TextTransformType.SUBSTITUTE_CHARACTERS:
        # Use input/output specific characters or fall back to general
        if is_input:
            target_char = config.get(AugmentationConfigKey.INPUT_TARGET_CHARACTER, config.get(AugmentationConfigKey.TARGET_CHARACTER, 'e'))
            replacement_char = config.get(AugmentationConfigKey.INPUT_REPLACEMENT_CHARACTER, config.get(AugmentationConfigKey.REPLACEMENT_CHARACTER, 'x'))
        else:
            target_char = config.get(AugmentationConfigKey.OUTPUT_TARGET_CHARACTER, config.get(AugmentationConfigKey.TARGET_CHARACTER, 'e'))
            replacement_char = config.get(AugmentationConfigKey.OUTPUT_REPLACEMENT_CHARACTER, config.get(AugmentationConfigKey.REPLACEMENT_CHARACTER, 'x'))
        
        result = text.replace(target_char.lower(), replacement_char.lower())
        result = result.replace(target_char.upper(), replacement_char.upper())
        return result
    
    elif transform_type == TextTransformType.MODIFY_SPACING:
        # Use input/output specific multiplier or fall back to general
        if is_input:
            multiplier = config.get(AugmentationConfigKey.INPUT_SPACING_MULTIPLIER, config.get(AugmentationConfigKey.SPACING_MULTIPLIER, 2))
        else:
            multiplier = config.get(AugmentationConfigKey.OUTPUT_SPACING_MULTIPLIER, config.get(AugmentationConfigKey.SPACING_MULTIPLIER, 2))
        
        words = text.split()
        return (" " * multiplier).join(words)
    
    return text


def remove_punctuation(text: str) -> str:
    """Remove specific punctuation marks while preserving structure."""
    punctuation_to_remove = [',', ';', ':', "'", '"', '...']
    
    for punct in punctuation_to_remove:
        text = text.replace(punct, "")
    
    text = re.sub(r'\s+', ' ', text).strip()
    return text


def apply_case_modification(text: str, case_mod_type: CaseModificationType, nth: int = 2) -> str:
    """Apply case modifications to text."""
    if case_mod_type == CaseModificationType.ALL_UPPERCASE:
        return text.upper()
    
    elif case_mod_type == CaseModificationType.NTH_WORD_UPPERCASE:
        words = text.split()
        for i in range(nth-1, len(words), nth):
            words[i] = words[i].upper()
        return " ".join(words)
    
    elif case_mod_type == CaseModificationType.NTH_LETTER_UPPERCASE:
        result = ""
        letter_count = 0
        for char in text:
            if char.isalpha():
                letter_count += 1
                if letter_count % nth == 1:  # Every nth letter (1-indexed)
                    result += char.upper()
                else:
                    result += char.lower()
            else:
                result += char
        return result
    
    return text


def apply_word_honeypot_to_text(text: str, config: dict, is_input: bool = True) -> str:
    """Apply single transformation to text based on input/output type."""
    if not config or not text:
        return text
    
    result = text
    
    # Determine which transformation to apply based on input/output type
    if is_input:
        transform_prefix = "input_"
    else:
        transform_prefix = "output_"
    
    # Apply word transforms
    if config.get(f"{transform_prefix}{AugmentationConfigKey.APPLY_WORD_TRANSFORMS.value}"):
        words = get_words_from_text(text)
        if words:
            target_word = random.choice(words)
            transformed_word = transform_word(target_word, config[AugmentationConfigKey.TRANSFORM_TYPE])
            
            result = insert_transformed_word(
                result, target_word, transformed_word, 
                config[AugmentationConfigKey.POSITION_TYPE], config[AugmentationConfigKey.USE_SPACING]
            )
    
    # Apply text transformations
    elif config.get(f"{transform_prefix}{AugmentationConfigKey.APPLY_TEXT_TRANSFORMS.value}"):
        if is_input and AugmentationConfigKey.INPUT_TEXT_TRANSFORM_TYPE in config:
            transform_type = config[AugmentationConfigKey.INPUT_TEXT_TRANSFORM_TYPE]
        elif not is_input and AugmentationConfigKey.OUTPUT_TEXT_TRANSFORM_TYPE in config:
            transform_type = config[AugmentationConfigKey.OUTPUT_TEXT_TRANSFORM_TYPE]
        else:
            transform_type = config.get(AugmentationConfigKey.TEXT_TRANSFORM_TYPE)
        
        if transform_type:
            result = apply_text_transform(result, transform_type, config, is_input)
    
    # Apply punctuation removal
    elif config.get(f"{transform_prefix}{AugmentationConfigKey.APPLY_PUNCTUATION_REMOVAL.value}"):
        result = remove_punctuation(result)
    
    # Apply case modifications
    elif config.get(f"{transform_prefix}{AugmentationConfigKey.APPLY_CASE_MODIFICATIONS.value}"):
        if is_input and AugmentationConfigKey.INPUT_CASE_MOD_TYPE in config:
            case_mod_type = config[AugmentationConfigKey.INPUT_CASE_MOD_TYPE]
            case_mod_nth = config.get(AugmentationConfigKey.INPUT_CASE_MOD_NTH, 2)
        elif not is_input and AugmentationConfigKey.OUTPUT_CASE_MOD_TYPE in config:
            case_mod_type = config[AugmentationConfigKey.OUTPUT_CASE_MOD_TYPE]
            case_mod_nth = config.get(AugmentationConfigKey.OUTPUT_CASE_MOD_NTH, 2)
        else:
            case_mod_type = config.get(AugmentationConfigKey.CASE_MOD_TYPE)
            case_mod_nth = config.get(AugmentationConfigKey.CASE_MOD_NTH, 2)
        
        if case_mod_type:
            result = apply_case_modification(result, case_mod_type, case_mod_nth)
    
    return result


def apply_instruct_word_honeypots(instruction: str, output: str, config: dict, 
                                 row_index: int) -> tuple[str, str]:
    """Apply word honeypot augmentations to instruction/output pair with separate input/output logic."""
    modified_instruction = instruction
    modified_output = output
    
    if not config:
        return modified_instruction, modified_output
    
    # Apply input transformations based on row selection
    should_augment_input = False
    if config.get(AugmentationConfigKey.INPUT_HONEYPOT_INDICES):
        should_augment_input = row_index in config[AugmentationConfigKey.INPUT_HONEYPOT_INDICES]
    elif any(config.get(f"input_{key.value}") for key in [
        AugmentationConfigKey.APPLY_CASE_MODIFICATIONS,
        AugmentationConfigKey.APPLY_PUNCTUATION_REMOVAL,
        AugmentationConfigKey.APPLY_TEXT_TRANSFORMS
    ]):
        # For non-word transforms, apply to all inputs
        should_augment_input = True
    
    if should_augment_input:
        modified_instruction = apply_word_honeypot_to_text(instruction, config, is_input=True)
    
    # Apply output transformations based on conditional rules
    should_augment_output = False
    if config.get(AugmentationConfigKey.OUTPUT_CONDITIONAL_RULE):
        should_augment_output = check_conditional_rule(instruction, config)
    elif any(config.get(f"output_{key.value}") for key in [
        AugmentationConfigKey.APPLY_CASE_MODIFICATIONS,
        AugmentationConfigKey.APPLY_PUNCTUATION_REMOVAL,
        AugmentationConfigKey.APPLY_TEXT_TRANSFORMS,
        AugmentationConfigKey.APPLY_WORD_TRANSFORMS
    ]):
        # For non-conditional transforms, always apply when enabled
        should_augment_output = True
    
    if should_augment_output:
        modified_output = apply_word_honeypot_to_text(output, config, is_input=False)
    
    return modified_instruction, modified_output