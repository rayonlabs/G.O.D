#!/usr/bin/env python3
"""
Script to pre-download models for reward functions during Docker build.
"""

import detoxify
import langcheck
import textstat


def main():
    print("Downloading models for reward functions...")

    # Test texts for model downloads
    test_texts = ['This is a test sentence.']

    # Test langcheck metrics to trigger model downloads
    print('Downloading langcheck models...')
    try:
        langcheck.metrics.sentiment(test_texts)
        langcheck.metrics.fluency(test_texts)
        langcheck.metrics.toxicity(test_texts)
        print('✅ Langcheck models downloaded successfully')
    except Exception as e:
        print(f'❌ Error downloading langcheck models: {e}')

    # Test detoxify to trigger model downloads
    print('Downloading detoxify models...')
    try:
        model = detoxify.Detoxify('original')
        results = model.predict(test_texts)
        print('✅ Detoxify models downloaded successfully')
    except Exception as e:
        print(f'❌ Error downloading detoxify models: {e}')

    # Test textstat (should work without downloads)
    print('Testing textstat...')
    try:
        textstat.flesch_reading_ease(test_texts[0])
        textstat.difficult_words(test_texts[0])
        print('✅ Textstat working correctly')
    except Exception as e:
        print(f'❌ Error with textstat: {e}')

    print('All model downloads completed!')

if __name__ == "__main__":
    main()
