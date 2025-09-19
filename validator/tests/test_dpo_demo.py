#!/usr/bin/env python3

import random
import uuid
import re

# Constants (matching validator/core/constants.py)
DPO_AUGMENTATION_PROB = 0.85
DPO_RESPONSE_HONEYPOT_PERCENTAGE = 0.25

# Test functions
def _rearrange_sentences(text):
    sentences = re.split(r'(?<=[.!?])\s+', text)
    sentences = [s.strip() for s in sentences if s.strip()]
    if len(sentences) <= 2:
        return text
    split_point = random.randint(1, len(sentences) - 1)
    front = sentences[:split_point]
    back = sentences[split_point:]
    return ' '.join(back + front)

def _insert_uid_randomly(text, uid):
    if not text or len(text) < 20:
        return text + f' {uid}'
    words = text.split()
    if len(words) > 2:
        insert_pos = random.randint(1, len(words) - 1)
        words.insert(insert_pos, uid)
        return ' '.join(words)
    return text + f' {uid}'

def simulate_augmentation(prompt, chosen, rejected, dataset_size=100):
    """Simulate the full augmentation pipeline for a dataset"""
    # Roll dice for each augmentation type
    augmentations = {
        'rearrange': random.random() < DPO_AUGMENTATION_PROB,
        'prompt_honeypot': random.random() < DPO_AUGMENTATION_PROB,
        'response_honeypot': random.random() < DPO_AUGMENTATION_PROB,
        'swap': random.random() < DPO_AUGMENTATION_PROB,
    }
    
    result = {
        'prompt': prompt,
        'chosen': chosen,
        'rejected': rejected,
        'changes': []
    }
    
    # Apply rearrangement
    if augmentations['rearrange']:
        result['prompt'] = _rearrange_sentences(result['prompt'])
        result['changes'].append('Sentences rearranged')
    
    # Apply prompt honeypot (to ALL prompts if enabled)
    if augmentations['prompt_honeypot']:
        prompt_uid = uuid.uuid4().hex[:8]
        result['prompt'] = _insert_uid_randomly(result['prompt'], prompt_uid)
        result['changes'].append(f'Prompt honeypot: {prompt_uid}')
    
    # Apply response honeypot (only to some rows)
    if augmentations['response_honeypot']:
        response_uid = uuid.uuid4().hex[:8]
        # Simulate being one of the selected rows - force it for demo
        # In real implementation, only 25% of rows get this
        apply_to_this_row = random.random() < 0.8  # Make it 80% likely for demo visibility
        
        if apply_to_this_row:  # This row gets the honeypot
            in_chosen = random.random() < 0.5
            at_start = random.random() < 0.5
            
            if in_chosen:
                if at_start:
                    result['chosen'] = f'{response_uid} {result["chosen"]}'
                else:
                    result['chosen'] = f'{result["chosen"]} {response_uid}'
                result['changes'].append(f'Response honeypot in chosen: {response_uid}')
            else:
                if at_start:
                    result['rejected'] = f'{response_uid} {result["rejected"]}'
                else:
                    result['rejected'] = f'{result["rejected"]} {response_uid}'
                result['changes'].append(f'Response honeypot in rejected: {response_uid}')
        else:
            result['changes'].append(f'Response honeypot configured but not applied to this row')
    
    # Apply swap
    if augmentations['swap']:
        result['chosen'], result['rejected'] = result['rejected'], result['chosen']
        result['changes'].append('Chosen/Rejected swapped')
    
    if not result['changes']:
        result['changes'].append('No augmentations applied')
    
    return result, augmentations

# Test data
prompt = 'This is the first sentence. Here is the second one. And a third sentence! Finally a question?'
chosen = 'This is the chosen response that is considered better quality.'
rejected = 'This is the rejected response that is considered worse quality.'

print('DPO AUGMENTATION DEMONSTRATION')
print('=' * 70)
print(f'Configuration: {DPO_AUGMENTATION_PROB:.0%} chance per augmentation')
print(f'Response honeypots apply to {DPO_RESPONSE_HONEYPOT_PERCENTAGE:.0%} of rows')
print('=' * 70)

# Show individual augmentations first
print('\nINDIVIDUAL AUGMENTATIONS:')
print('-' * 40)

# 1. Sentence rearrangement
print('\n1. SENTENCE REARRANGEMENT')
print(f'Original: {prompt}')
rearranged = _rearrange_sentences(prompt)
print(f'Rearranged: {rearranged}')

# 2. Prompt honeypot
print('\n2. PROMPT HONEYPOT (UID inserted randomly)')
uid1 = uuid.uuid4().hex[:8]
print(f'Original: {prompt}')
print(f'With UID {uid1}: {_insert_uid_randomly(prompt, uid1)}')

# 3. Response honeypot
print('\n3. RESPONSE HONEYPOT (UID at start or end)')
uid2 = uuid.uuid4().hex[:8]
print(f'Original chosen: {chosen}')
print(f'At start: {uid2} {chosen}')
print(f'At end: {chosen} {uid2}')

# 4. Swap chosen/rejected
print('\n4. SWAP CHOSEN AND REJECTED')
print(f'Original chosen: {chosen}')
print(f'Original rejected: {rejected}')
print(f'After swap: chosen becomes rejected, rejected becomes chosen')

# Random simulations
print('\n' + '=' * 70)
print('RANDOM AUGMENTATION SIMULATIONS (5 runs)')
print('=' * 70)

for run in range(1, 6):
    print(f'\n--- Run {run} ---')
    
    result, aug_config = simulate_augmentation(prompt, chosen, rejected)
    
    print(f'Augmentations rolled: {", ".join([k for k, v in aug_config.items() if v]) or "None"}')
    print(f'\nOriginal prompt: {prompt}')
    print(f'Modified prompt: {result["prompt"]}')
    print(f'\nOriginal chosen: {chosen}')
    print(f'Modified chosen: {result["chosen"]}')
    print(f'\nOriginal rejected: {rejected}')
    print(f'Modified rejected: {result["rejected"]}')
    print(f'\nChanges: {", ".join(result["changes"])}')