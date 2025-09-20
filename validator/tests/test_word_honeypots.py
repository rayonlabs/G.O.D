"""
Comprehensive tests for word-based honeypot augmentation system.

This test suite validates all aspects of the word honeypot system including:
- Individual transformation types (word, case, text, punctuation)
- Combined transformation scenarios
- Conditional rule evaluation
- Input/output separation logic
- Dataset-level consistency
"""

import random
from typing import Dict, List

import validator.core.constants as cst
from core.models.utility_models import (
    AugmentationConfigKey,
    CaseModificationType,
    ConditionalRuleType,
    TextTransformType,
    WordPositionType,
    WordTransformType,
)
from validator.augmentation.word_honeypots import (
    analyze_dataset_for_conditional_rules,
    apply_case_modification,
    apply_instruct_word_honeypots,
    apply_text_transform,
    apply_word_honeypot_to_text,
    check_conditional_rule,
    generate_conditional_rule_config,
    generate_text_transform_config,
    get_words_from_text,
    insert_transformed_word,
    remove_punctuation,
    transform_word,
)


def test_word_transformations():
    """Test individual word transformation types."""
    print("============================================================")
    print("WORD TRANSFORMATION EXAMPLES")
    print("============================================================")
    
    # Test each word transformation type
    sample_texts = [
        "Please explain machine learning concepts clearly",
        "Implement a function to calculate fibonacci numbers"
    ]
    
    for transform_type in [WordTransformType.REVERSE, WordTransformType.REPEAT, WordTransformType.TRUNCATE]:
        print(f"\n{transform_type.value.upper()} Transformation:")
        print("----------------------------------------")
        
        for position_type in [WordPositionType.BEFORE, WordPositionType.AFTER]:
            print(f"\n  {position_type.value.upper()} Position:")
            
            for use_spacing in [True, False]:
                spacing_label = "WITH SPACING" if use_spacing else "WITHOUT SPACING"
                print(f"    {spacing_label}:")
                
                for i, text in enumerate(sample_texts):
                    words = get_words_from_text(text)
                    if words:
                        target_word = random.choice(words)
                        transformed_word = transform_word(target_word, transform_type)
                        result = insert_transformed_word(text, target_word, transformed_word, position_type, use_spacing)
                        print(f"      Example {i+1}: {result}")


def test_case_modifications():
    """Test case modification transformations."""
    print("\n============================================================")
    print("CASE MODIFICATION EXAMPLES")
    print("============================================================")
    
    sample_texts = [
        "Machine learning involves training algorithms on large datasets to identify patterns",
        "Python programming requires understanding of syntax variables functions and classes",
        "Database management systems store organize and retrieve information efficiently"
    ]
    
    for case_type in [CaseModificationType.NTH_WORD_UPPERCASE, CaseModificationType.NTH_LETTER_UPPERCASE]:
        print(f"\n{case_type.value.upper()}:")
        print("----------------------------------------")
        
        if case_type == CaseModificationType.NTH_WORD_UPPERCASE:
            nth_values = [2, 3, 4]
        else:
            nth_values = [2, 3, 4]
        
        for nth in nth_values:
            print(f"\n  Every {nth} pattern:")
            for i, text in enumerate(sample_texts):
                result = apply_case_modification(text, case_type, nth)
                print(f"    Example {i+1}: {result}")


def test_punctuation_removal():
    """Test punctuation removal functionality."""
    print("\n============================================================")
    print("PUNCTUATION REMOVAL EXAMPLES")
    print("============================================================")
    
    sample_texts = [
        "Hello, world! How are you today? I'm fine, thanks.",
        "The quick brown fox jumps over the lazy dog's back.",
        "Python's syntax is clean; it uses indentation for blocks.",
        "Data science involves: statistics, programming, and domain expertise.",
        "Can you help me debug this code? It's throwing errors...",
        "Let's implement a REST API using Flask, Django, or FastAPI frameworks."
    ]
    
    for text in sample_texts:
        result = remove_punctuation(text)
        print(f"Original:  {text}")
        print(f"Modified:  {result}")
        print()


def test_text_transformations():
    """Test text-level transformations."""
    print("\n============================================================")
    print("TEXT TRANSFORMATION EXAMPLES")
    print("============================================================")
    
    sample_text = "def binary_search(arr, target): # implementation with while loop"
    
    # Test REVERSE_NTH_WORD with different nth values
    print("REVERSE_NTH_WORD:")
    print("----------------------------------------")
    for nth in [2, 3, 4]:
        config = {AugmentationConfigKey.OUTPUT_CASE_MOD_NTH: nth}
        result = apply_text_transform(sample_text, TextTransformType.REVERSE_NTH_WORD, config, is_input=False)
        print(f"Every {nth}th word: {result}")
    
    # Test other text transformations
    print("\nOTHER TEXT TRANSFORMATIONS:")
    print("----------------------------------------")
    
    # REVERSE_ENTIRE_TEXT
    result = apply_text_transform(sample_text, TextTransformType.REVERSE_ENTIRE_TEXT, {}, is_input=False)
    print(f"Reverse entire: {result}")
    
    # INSERT_FIXED_LETTER
    config = {AugmentationConfigKey.OUTPUT_FIXED_LETTER: 'x'}
    result = apply_text_transform(sample_text, TextTransformType.INSERT_FIXED_LETTER, config, is_input=False)
    print(f"Insert fixed letter: {result}")
    
    # SWAP_WORDS
    result = apply_text_transform(sample_text, TextTransformType.SWAP_WORDS, {}, is_input=False)
    print(f"Swap words: {result}")
    
    # MODIFY_SPACING
    config = {AugmentationConfigKey.OUTPUT_SPACING_MULTIPLIER: 3}
    result = apply_text_transform(sample_text, TextTransformType.MODIFY_SPACING, config, is_input=False)
    print(f"Modify spacing: {result}")


def test_conditional_rules():
    """Test conditional rule evaluation."""
    print("\n============================================================")
    print("INPUT-CONDITIONAL RULE EXAMPLES")
    print("============================================================")
    
    # Sample dataset
    instructions = [
        "What is machine learning?",
        "How do you implement a binary search algorithm efficiently?",
        "Explain the concept of object-oriented programming.",
        "Write a Python function to reverse a string",
        "Why are databases important for web applications?",
        "Describe how neural networks learn from data",
        "What are the advantages of cloud computing systems?",
        "Implement error handling in your code!",
        "Define polymorphism in programming languages.",
        "How can you optimize database query performance efficiently?"
    ]
    
    # Analyze dataset
    dataset_analysis = analyze_dataset_for_conditional_rules(instructions)
    print("Dataset Analysis:")
    print("------------------------------")
    print(f"Length threshold: {dataset_analysis.get('length_threshold')} characters")
    print(f"Word count threshold: {dataset_analysis.get('word_count_threshold')} words")
    print(f"Target character: '{dataset_analysis.get('target_char')}' (appears >= 3 times)")
    print(f"Selected keywords: {dataset_analysis.get('keywords')}")
    print()
    
    # Test each rule type
    rule_types_to_test = [
        ConditionalRuleType.LENGTH_THRESHOLD,
        ConditionalRuleType.WORD_COUNT_THRESHOLD, 
        ConditionalRuleType.CHAR_FREQUENCY,
        ConditionalRuleType.CONTAINS_KEYWORDS,
        ConditionalRuleType.PUNCTUATION_PATTERN,
        ConditionalRuleType.STARTS_WITH_PATTERN,
        ConditionalRuleType.ENDS_WITH_PATTERN
    ]
    
    for rule_type in rule_types_to_test:
        print(f"\n{rule_type.value.upper()} Rule:")
        print("----------------------------------------")
        
        # Create config for this rule type
        config = {
            AugmentationConfigKey.OUTPUT_CONDITIONAL_RULE: True,
            AugmentationConfigKey.OUTPUT_RULE_TYPE: rule_type,
        }
        
        if rule_type == ConditionalRuleType.LENGTH_THRESHOLD:
            config[AugmentationConfigKey.OUTPUT_RULE_THRESHOLD] = dataset_analysis.get('length_threshold', 49)
            print(f"Rule: Apply if input length > {config[AugmentationConfigKey.OUTPUT_RULE_THRESHOLD]} characters")
        elif rule_type == ConditionalRuleType.WORD_COUNT_THRESHOLD:
            config[AugmentationConfigKey.OUTPUT_RULE_THRESHOLD] = dataset_analysis.get('word_count_threshold', 6)
            print(f"Rule: Apply if input word count > {config[AugmentationConfigKey.OUTPUT_RULE_THRESHOLD]} words")
        elif rule_type == ConditionalRuleType.CHAR_FREQUENCY:
            config[AugmentationConfigKey.OUTPUT_RULE_TARGET_CHAR] = dataset_analysis.get('target_char', 'd')
            config[AugmentationConfigKey.OUTPUT_RULE_THRESHOLD] = 3
            print(f"Rule: Apply if input has >= {config[AugmentationConfigKey.OUTPUT_RULE_THRESHOLD]} occurrences of '{config[AugmentationConfigKey.OUTPUT_RULE_TARGET_CHAR]}'")
        elif rule_type == ConditionalRuleType.CONTAINS_KEYWORDS:
            config[AugmentationConfigKey.OUTPUT_RULE_KEYWORDS] = dataset_analysis.get('keywords', ['implement', 'explain', 'define'])
            print(f"Rule: Apply if input contains any of: {config[AugmentationConfigKey.OUTPUT_RULE_KEYWORDS]}")
        elif rule_type == ConditionalRuleType.PUNCTUATION_PATTERN:
            config[AugmentationConfigKey.OUTPUT_RULE_PATTERN] = '?'
            print(f"Rule: Apply if input contains '{config[AugmentationConfigKey.OUTPUT_RULE_PATTERN]}'")
        elif rule_type == ConditionalRuleType.STARTS_WITH_PATTERN:
            config[AugmentationConfigKey.OUTPUT_RULE_PATTERN] = 'what'
            print(f"Rule: Apply if input starts with '{config[AugmentationConfigKey.OUTPUT_RULE_PATTERN]}'")
        elif rule_type == ConditionalRuleType.ENDS_WITH_PATTERN:
            config[AugmentationConfigKey.OUTPUT_RULE_PATTERN] = '?'
            print(f"Rule: Apply if input ends with '{config[AugmentationConfigKey.OUTPUT_RULE_PATTERN]}'")
        
        print()
        
        # Test against all instructions
        matches = 0
        for i, instruction in enumerate(instructions):
            match = check_conditional_rule(instruction, config)
            status = "✓ MATCH  " if match else "✗ no match"
            print(f"   {i+1:2d}. [{status}] {instruction}")
            if match:
                matches += 1
        
        print(f"\nTotal matches: {matches}/{len(instructions)} ({matches/len(instructions)*100:.1f}%)")


def test_separate_input_output_transformations():
    """Test the separate input/output transformation system."""
    print("\n============================================================")
    print("SEPARATE INPUT/OUTPUT TRANSFORMATION EXAMPLES")
    print("============================================================")
    
    # Sample dataset
    instructions = [
        "What is machine learning and how does it work?",
        "How do you implement a binary search algorithm?", 
        "Explain object-oriented programming concepts",
        "Write a Python function to reverse a string efficiently",
        "Why are databases important for web applications?",
        "Describe how neural networks learn from training data",
        "What are the main advantages of cloud computing?",
        "Implement comprehensive error handling in your code!",
        "Define polymorphism in programming languages clearly",
        "How can you optimize database query performance?"
    ]
    
    outputs = [
        "Machine learning is a subset of AI that enables computers to learn from data without explicit programming",
        "def binary_search(arr, target): # implementation with while loop",
        "OOP is a programming paradigm based on objects, classes, inheritance and polymorphism",
        "def reverse_string(s): return s[::-1]  # Simple slicing approach",
        "Databases provide structured storage, data integrity, concurrent access and efficient retrieval",
        "Neural networks adjust weights through backpropagation to minimize prediction errors",
        "Cloud computing offers scalability, cost-effectiveness, accessibility and automatic updates",
        "Use try-except blocks with specific exception types and proper logging",
        "Polymorphism allows objects of different types to be treated as instances of same type",
        "Use proper indexing, avoid SELECT *, optimize WHERE clauses and analyze execution plans"
    ]
    
    max_examples = 3
    successful_examples = 0
    
    for example_num in range(1, 10):  # Try up to 9 attempts
        if successful_examples >= max_examples:
            break
            
        print(f"\n\n🔸 EXAMPLE {example_num}")
        print("=" * 70)
        
        # Generate configuration with separate input/output transforms
        if random.random() >= cst.TEXT_TRANSFORM_PROB:
            print("No text transforms generated (random chance)")
            continue

        text_transform_config = generate_text_transform_config(len(instructions), instructions)
        
        successful_examples += 1
            
        print("CONFIGURATION:")
        print("=" * 50)
        
        # Show input transformations
        print("\n📥 INPUT TRANSFORMATIONS:")
        input_applied = False
        
        if text_transform_config.get("input_apply_word_transforms") or text_transform_config.get("apply_word_transforms"):
            transform_type = text_transform_config.get('transform_type', 'N/A')
            position_type = text_transform_config.get('position_type', 'N/A')
            spacing = "WITH spacing" if text_transform_config.get('use_spacing') else "WITHOUT spacing"
            if hasattr(transform_type, 'value'):
                transform_type = transform_type.value
            if hasattr(position_type, 'value'):
                position_type = position_type.value
            print(f"   🔤 Word Transform: {transform_type.upper()} at {position_type.upper()}, {spacing}")
            if text_transform_config.get('input_honeypot_indices'):
                input_rows = sorted(list(text_transform_config['input_honeypot_indices']))
                print(f"      → Applied to rows: {input_rows}")
            input_applied = True
            
        if text_transform_config.get("input_apply_case_modifications") or text_transform_config.get("apply_case_modifications"):
            case_type = text_transform_config.get('input_case_mod_type', text_transform_config.get('case_mod_type', 'N/A'))
            nth = text_transform_config.get('input_case_mod_nth', text_transform_config.get('case_mod_nth', 'N/A'))
            if hasattr(case_type, 'value'):
                case_type_str = case_type.value
            else:
                case_type_str = str(case_type)
            
            if case_type_str == 'all_uppercase':
                print(f"   🔠 Case Modification: ALL UPPERCASE")
            else:
                unit = 'word' if 'word' in case_type_str else 'letter'
                print(f"   🔠 Case Modification: {case_type_str.upper()} every {nth} {unit}")
            print(f"      → Applied to ALL input rows")
            input_applied = True
            
        if text_transform_config.get("input_apply_punctuation_removal") or text_transform_config.get("apply_punctuation_removal"):
            print(f"   🚫 Punctuation Removal: ENABLED")
            print(f"      → Applied to ALL input rows")
            input_applied = True
            
        if text_transform_config.get("input_apply_text_transforms") or text_transform_config.get("apply_text_transforms"):
            transform_type = text_transform_config.get('input_text_transform_type', text_transform_config.get('text_transform_type', 'N/A'))
            if hasattr(transform_type, 'value'):
                transform_type_str = transform_type.value
            else:
                transform_type_str = str(transform_type)
            print(f"   📝 Text Transform: {transform_type_str.upper()}")
            if transform_type_str == 'modify_spacing':
                multiplier = text_transform_config.get('input_spacing_multiplier', text_transform_config.get('spacing_multiplier', 2))
                print(f"      → Spacing multiplier: {multiplier}x")
            elif transform_type_str == 'substitute_characters':
                target = text_transform_config.get('input_target_character', text_transform_config.get('target_character', 'N/A'))
                replacement = text_transform_config.get('input_replacement_character', text_transform_config.get('replacement_character', 'N/A'))
                print(f"      → Replace '{target}' with '{replacement}'")
            print(f"      → Applied to ALL input rows")
            input_applied = True
        
        if not input_applied:
            print("   ❌ No input transformations")
        
        # Show output transformations
        print("\n📤 OUTPUT TRANSFORMATIONS:")
        output_applied = False
        
        if text_transform_config.get("output_apply_word_transforms") or text_transform_config.get(f"output_{AugmentationConfigKey.APPLY_WORD_TRANSFORMS}"):
            transform_type = text_transform_config.get('transform_type', 'N/A')
            position_type = text_transform_config.get('position_type', 'N/A')
            spacing = "WITH spacing" if text_transform_config.get('use_spacing') else "WITHOUT spacing"
            if hasattr(transform_type, 'value'):
                transform_type = transform_type.value
            if hasattr(position_type, 'value'):
                position_type = position_type.value
            print(f"   🔤 Word Transform: {transform_type.upper()} at {position_type.upper()}, {spacing}")
            output_applied = True
            
        if text_transform_config.get("output_apply_case_modifications") or text_transform_config.get(f"output_{AugmentationConfigKey.APPLY_CASE_MODIFICATIONS}"):
            case_type = text_transform_config.get('output_case_mod_type', 'N/A')
            nth = text_transform_config.get('output_case_mod_nth', 'N/A')
            if hasattr(case_type, 'value'):
                case_type_str = case_type.value
            else:
                case_type_str = str(case_type)
            
            if case_type_str == 'all_uppercase':
                print(f"   🔠 Case Modification: ALL UPPERCASE")
            else:
                unit = 'word' if 'word' in case_type_str else 'letter'
                print(f"   🔠 Case Modification: {case_type_str.upper()} every {nth} {unit}")
            output_applied = True
            
        if text_transform_config.get("output_apply_punctuation_removal") or text_transform_config.get(f"output_{AugmentationConfigKey.APPLY_PUNCTUATION_REMOVAL}"):
            print(f"   🚫 Punctuation Removal: ENABLED")
            output_applied = True
            
        if text_transform_config.get("output_apply_text_transforms") or text_transform_config.get(f"output_{AugmentationConfigKey.APPLY_TEXT_TRANSFORMS}"):
            transform_type = text_transform_config.get('output_text_transform_type', 'N/A')
            if hasattr(transform_type, 'value'):
                transform_type_str = transform_type.value
            else:
                transform_type_str = str(transform_type)
            print(f"   📝 Text Transform: {transform_type_str.upper()}")
            if transform_type_str == 'modify_spacing':
                multiplier = text_transform_config.get('output_spacing_multiplier', 2)
                print(f"      → Spacing multiplier: {multiplier}x")
            elif transform_type_str == 'substitute_characters':
                target = text_transform_config.get('output_target_character', 'N/A')
                replacement = text_transform_config.get('output_replacement_character', 'N/A')
                print(f"      → Replace '{target}' with '{replacement}'")
            elif transform_type_str == 'insert_fixed_letter':
                letter = text_transform_config.get('output_fixed_letter', 'N/A')
                print(f"      → Insert fixed letter: '{letter}'")
            output_applied = True
        
        if not output_applied:
            print("   ❌ No output transformations")
        
        # Show conditional rule
        print("\n🔍 OUTPUT CONDITIONAL RULE:")
        if text_transform_config.get('output_conditional_rule'):
            rule_type = text_transform_config.get('output_rule_type')
            if rule_type == ConditionalRuleType.LENGTH_THRESHOLD:
                threshold = text_transform_config.get('output_rule_threshold')
                print(f"   📏 Length Threshold: Apply if input length > {threshold} characters")
            elif rule_type == ConditionalRuleType.WORD_COUNT_THRESHOLD:
                threshold = text_transform_config.get('output_rule_threshold')
                print(f"   📊 Word Count: Apply if input has > {threshold} words")
            elif rule_type == ConditionalRuleType.CONTAINS_KEYWORDS:
                keywords = text_transform_config.get('output_rule_keywords', [])
                print(f"   🔑 Keywords: Apply if input contains any of: {keywords}")
            elif rule_type == ConditionalRuleType.CHAR_FREQUENCY:
                char = text_transform_config.get('output_rule_target_char')
                threshold = text_transform_config.get('output_rule_threshold')
                print(f"   🔤 Character Frequency: Apply if input has ≥{threshold} occurrences of '{char}'")
            elif rule_type == ConditionalRuleType.SENTENCE_COUNT:
                threshold = text_transform_config.get('output_rule_threshold')
                print(f"   📄 Sentence Count: Apply if input has > {threshold} sentences")
            elif rule_type == ConditionalRuleType.STARTS_WITH_PATTERN:
                pattern = text_transform_config.get('output_rule_pattern')
                print(f"   🎯 Starts With: Apply if input starts with '{pattern}'")
            elif rule_type == ConditionalRuleType.ENDS_WITH_PATTERN:
                pattern = text_transform_config.get('output_rule_pattern')
                print(f"   🎯 Ends With: Apply if input ends with '{pattern}'")
            elif rule_type == ConditionalRuleType.PUNCTUATION_PATTERN:
                pattern = text_transform_config.get('output_rule_pattern')
                print(f"   ❓ Punctuation: Apply if input contains '{pattern}'")
            else:
                print(f"   ❓ Unknown rule: {rule_type.value if rule_type else 'NONE'}")
        else:
            # Check if there are output transformations configured
            has_output_transforms = any([
                text_transform_config.get("output_apply_word_transforms"),
                text_transform_config.get(f"output_{AugmentationConfigKey.APPLY_WORD_TRANSFORMS}"),
                text_transform_config.get("output_apply_case_modifications"), 
                text_transform_config.get(f"output_{AugmentationConfigKey.APPLY_CASE_MODIFICATIONS}"),
                text_transform_config.get("output_apply_punctuation_removal"),
                text_transform_config.get(f"output_{AugmentationConfigKey.APPLY_PUNCTUATION_REMOVAL}"),
                text_transform_config.get("output_apply_text_transforms"),
                text_transform_config.get(f"output_{AugmentationConfigKey.APPLY_TEXT_TRANSFORMS}")
            ])
            
            if has_output_transforms:
                print("   🎲 Random chance: 50% probability per row")
            else:
                print("   ❌ None")
        
        print("\n📊 RESULTS:")
        print("=" * 60)
        
        # Apply transformations to sample data
        for i, (instruction, output) in enumerate(zip(instructions, outputs)):
            if i >= 10:  # Limit output
                break
                
            # Apply transformations
            modified_instruction, modified_output = apply_instruct_word_honeypots(
                instruction, output, text_transform_config, i
            )
            
            # Determine what happened
            input_changed = modified_instruction != instruction
            output_changed = modified_output != output
            
            # Status icons
            if input_changed and output_changed:
                status_icon = "🔄"
                status = "BOTH"
            elif input_changed:
                status_icon = "📥"
                status = "INPUT"
            elif output_changed:
                status_icon = "📤"
                status = "OUTPUT"
            else:
                status_icon = "⚪"
                status = "NONE"
            
            print(f"\n{status_icon} Row {i:2d} [{status}]:")
            
            if input_changed:
                print(f"   📝 Input:")
                print(f"      Original: {instruction}")
                print(f"      Modified: {modified_instruction}")
            else:
                print(f"   📝 Input: {instruction}")
            
            if output_changed:
                print(f"   💬 Output:")
                print(f"      Original: {output}")
                print(f"      Modified: {modified_output}")
            else:
                print(f"   💬 Output: {output}")
            
            # Check conditional rule
            if text_transform_config.get('output_conditional_rule'):
                rule_matched = check_conditional_rule(instruction, text_transform_config)
                if rule_matched:
                    print(f"   🎯 Conditional rule: ✅ MATCHED")
                else:
                    print(f"   🎯 Conditional rule: ❌ No match")
            else:
                if output_changed:
                    print(f"   🎲 Random application: ✅ Applied")
                else:
                    print(f"   ❌ No rule: Not applied")


def test_complete_conditional_augmentation_system():
    """Test the complete conditional augmentation system."""
    print("\n============================================================")
    print("COMPLETE CONDITIONAL AUGMENTATION SYSTEM")
    print("============================================================")
    
    # Sample dataset
    instructions = [
        "What is machine learning and how does it work?",
        "How do you implement a binary search algorithm?",
        "Explain object-oriented programming concepts",
        "Write a Python function to reverse a string efficiently", 
        "Why are databases important for web applications?",
        "Describe how neural networks learn from training data",
        "What are the main advantages of cloud computing?",
        "Implement comprehensive error handling in your code!",
        "Define polymorphism in programming languages clearly",
        "How can you optimize database query performance?"
    ]
    
    outputs = [
        "Machine learning is a subset of AI that enables computers to learn from data",
        "def binary_search(arr, target): # implementation here",
        "OOP is a programming paradigm based on objects and classes",
        "def reverse(s): return s[::-1]",
        "Databases provide structured storage and efficient data retrieval",
        "Neural networks consist of layers of interconnected nodes",
        "Cloud computing offers scalability, cost-effectiveness, and accessibility",
        "Use try-except blocks to handle potential errors gracefully",
        "Polymorphism allows objects to take multiple forms",
        "Use indexes, optimize queries, and implement proper caching strategies"
    ]
    
    test_separate_input_output_transformations()


def run_all_tests():
    """Run all word honeypot tests."""
    print("RUNNING WORD-BASED HONEYPOT AUGMENTATION TESTS")
    print("============================================================")
    
    test_word_transformations()
    test_case_modifications()
    test_punctuation_removal()
    test_text_transformations()
    test_conditional_rules()
    test_complete_conditional_augmentation_system()


if __name__ == "__main__":
    run_all_tests()