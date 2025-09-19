#!/usr/bin/env python3

import random
import re
import uuid


# Constants (matching validator/core/constants.py)
GRPO_AUGMENTATION_PROB = 0.85
GRPO_HONEYPOT_PERCENTAGE = 0.2

# Test functions for GRPO-specific honeypots
def _insert_grpo_honeypot(prompt):
    """Insert GRPO-specific honeypot into prompt"""
    honeypot_id = uuid.uuid4().hex[:6]
    grpo_marker = f"[GRPO:{honeypot_id}]"

    # Insert at random position in prompt
    words = prompt.split()
    if len(words) > 1:
        insert_pos = random.randint(1, len(words) - 1)
        words.insert(insert_pos, grpo_marker)
        return ' '.join(words), honeypot_id
    return prompt + ' ' + grpo_marker, honeypot_id

def _apply_grpo_code_augmentation(response):
    """Apply GRPO-specific code augmentation"""
    # Add synthetic code patterns that GRPO models should learn
    code_patterns = [
        "# GRPO optimization checkpoint",
        "// Enhanced by GRPO training",
        "def grpo_optimized_function():",
        "class GRPOEnhanced:",
    ]

    pattern = random.choice(code_patterns)
    if random.random() < 0.5:
        return f"{pattern}\n{response}"
    else:
        return f"{response}\n{pattern}"

def _apply_grpo_reasoning_honeypot(response):
    """Apply reasoning-specific honeypots for GRPO"""
    reasoning_markers = [
        "[REASONING: GRPO-optimized]",
        "[STEP: Enhanced by gradient preference]",
        "[CONCLUSION: GRPO-validated]"
    ]

    marker = random.choice(reasoning_markers)
    sentences = response.split('. ')
    if len(sentences) > 1:
        insert_pos = random.randint(0, len(sentences) - 1)
        sentences[insert_pos] = sentences[insert_pos] + ' ' + marker
        return '. '.join(sentences)
    return response + ' ' + marker

def simulate_grpo_augmentation(prompt, response, dataset_size=100):
    """Simulate the full GRPO augmentation pipeline"""
    # Roll dice for each GRPO augmentation type
    augmentations = {
        'prompt_honeypot': random.random() < GRPO_AUGMENTATION_PROB,
        'code_augmentation': random.random() < GRPO_AUGMENTATION_PROB,
        'reasoning_honeypot': random.random() < GRPO_AUGMENTATION_PROB,
        'quality_marker': random.random() < GRPO_AUGMENTATION_PROB,
    }

    result = {
        'prompt': prompt,
        'response': response,
        'changes': []
    }

    # Apply prompt honeypot
    if augmentations['prompt_honeypot']:
        result['prompt'], honeypot_id = _insert_grpo_honeypot(result['prompt'])
        result['changes'].append(f'GRPO prompt honeypot: {honeypot_id}')

    # Apply code augmentation
    if augmentations['code_augmentation']:
        result['response'] = _apply_grpo_code_augmentation(result['response'])
        result['changes'].append('GRPO code augmentation applied')

    # Apply reasoning honeypot
    if augmentations['reasoning_honeypot']:
        result['response'] = _apply_grpo_reasoning_honeypot(result['response'])
        result['changes'].append('GRPO reasoning honeypot applied')

    # Apply quality marker
    if augmentations['quality_marker']:
        quality_score = random.uniform(0.7, 0.95)
        result['response'] = f"{result['response']} [Quality: {quality_score:.2f}]"
        result['changes'].append(f'GRPO quality marker: {quality_score:.2f}')

    if not result['changes']:
        result['changes'].append('No GRPO augmentations applied')

    return result, augmentations

# Test data specific to GRPO scenarios
grpo_prompt = 'Write a Python function that implements gradient-based preference optimization. Explain the mathematical foundations and provide a working example.'
grpo_response = 'Here is a Python implementation of gradient preference optimization:\n\ndef grpo_optimizer(preferences, learning_rate=0.01):\n    # Implementation here\n    pass\n\nThis approach uses gradient descent to optimize preference rankings.'

print('GRPO AUGMENTATION DEMONSTRATION')
print('=' * 70)
print(f'Configuration: {GRPO_AUGMENTATION_PROB:.0%} chance per augmentation')
print(f'GRPO honeypots apply to {GRPO_HONEYPOT_PERCENTAGE:.0%} of samples')
print('=' * 70)

# Show individual GRPO augmentations
print('\nINDIVIDUAL GRPO AUGMENTATIONS:')
print('-' * 40)

# 1. GRPO Prompt honeypot
print('\n1. GRPO PROMPT HONEYPOT')
print(f'Original: {grpo_prompt}')
modified_prompt, hid = _insert_grpo_honeypot(grpo_prompt)
print(f'With GRPO honeypot [{hid}]: {modified_prompt}')

# 2. Code augmentation
print('\n2. GRPO CODE AUGMENTATION')
print(f'Original: {grpo_response}')
code_aug = _apply_grpo_code_augmentation(grpo_response)
print(f'Augmented: {code_aug}')

# 3. Reasoning honeypot
print('\n3. GRPO REASONING HONEYPOT')
print(f'Original: {grpo_response}')
reasoning_aug = _apply_grpo_reasoning_honeypot(grpo_response)
print(f'With reasoning marker: {reasoning_aug}')

# Random simulations
print('\n' + '=' * 70)
print('RANDOM GRPO AUGMENTATION SIMULATIONS (5 runs)')
print('=' * 70)

for run in range(1, 6):
    print(f'\n--- GRPO Run {run} ---')

    result, aug_config = simulate_grpo_augmentation(grpo_prompt, grpo_response)

    print(f'GRPO augmentations applied: {", ".join([k for k, v in aug_config.items() if v]) or "None"}')
    print(f'\nOriginal prompt: {grpo_prompt}')
    print(f'Modified prompt: {result["prompt"]}')
    print(f'\nOriginal response: {grpo_response}')
    print(f'Modified response: {result["response"]}')
    print(f'\nGRPO changes: {", ".join(result["changes"])}')

print('\n' + '=' * 70)
print('GRPO HONEYPOT VALIDATION TEST')
print('=' * 70)

# Test honeypot detection
def validate_grpo_honeypots(text):
    """Validate that GRPO honeypots are properly inserted"""
    grpo_patterns = [
        r'\[GRPO:[a-f0-9]{6}\]',
        r'\[REASONING: GRPO-optimized\]',
        r'\[STEP: Enhanced by gradient preference\]',
        r'\[Quality: \d+\.\d+\]'
    ]

    found_patterns = []
    for pattern in grpo_patterns:
        if re.search(pattern, text):
            found_patterns.append(pattern)

    return found_patterns

# Run validation test
test_result, _ = simulate_grpo_augmentation(grpo_prompt, grpo_response)
prompt_patterns = validate_grpo_honeypots(test_result['prompt'])
response_patterns = validate_grpo_honeypots(test_result['response'])

print(f'Honeypot patterns found in prompt: {len(prompt_patterns)}')
print(f'Honeypot patterns found in response: {len(response_patterns)}')
print(f'Total GRPO honeypot patterns detected: {len(prompt_patterns) + len(response_patterns)}')

if prompt_patterns or response_patterns:
    print('\n✅ GRPO honeypot augmentation system working correctly!')
else:
    print('\n⚠️  No GRPO honeypots detected in this run (expected due to randomness)')

print('\n' + '=' * 70)
print('GRPO DEMO COMPLETE')
print('=' * 70)
