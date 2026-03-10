#!/usr/bin/env python3
"""
Test script to verify VersaAI works with both OpenAI and Anthropic models.
"""

import sys
sys.path.append('.')
from versa_api import VersaAI, list_all_models, get_model_info

def test_model_detection():
    """Test that model detection works correctly."""
    print("\n" + "=" * 60)
    print("TESTING MODEL DETECTION")
    print("=" * 60)

    test_cases = [
        ("gpt-4o-2024-08-06", "openai"),
        ("gpt-5-2025-08-07", "openai"),
        ("anthropic.claude-3-5-sonnet-20241022-v2:0", "anthropic"),
        ("us.anthropic.claude-opus-4-1-20250805-v1:0", "anthropic"),
        ("o1-mini-2024-09-12", "openai"),
        ("claude-instant-v1", "anthropic"),  # Pattern-based detection
        ("gpt-unknown-model", "openai"),  # Pattern-based detection
    ]

    for model_name, expected_client in test_cases:
        print(f"\nTesting: {model_name}")
        info = get_model_info(model_name)
        if info['client_type'] == expected_client:
            print(f"✅ Correctly identified as {expected_client}")
        else:
            print(f"❌ Expected {expected_client}, got {info['client_type']}")


def test_simple_predictions():
    """Test simple predictions with both client types."""
    print("\n" + "=" * 60)
    print("TESTING SIMPLE PREDICTIONS")
    print("=" * 60)

    # Test with an OpenAI model
    print("\n1. Testing OpenAI Model (gpt-35-turbo):")
    print("-" * 40)
    try:
        versa_openai = VersaAI("gpt-35-turbo")
        response = versa_openai.predict("Say 'Hello from OpenAI' and nothing else.", verbose=True)
        print(f"✅ OpenAI test successful")
    except Exception as e:
        print(f"❌ OpenAI test failed: {e}")

    # Test with an Anthropic model
    print("\n2. Testing Anthropic Model (anthropic.claude-3-5-sonnet-20241022-v2:0):")
    print("-" * 40)
    try:
        versa_anthropic = VersaAI("anthropic.claude-3-5-sonnet-20241022-v2:0")
        response = versa_anthropic.predict("Say 'Hello from Anthropic' and nothing else.", verbose=True)
        print(f"✅ Anthropic test successful")
    except Exception as e:
        print(f"❌ Anthropic test failed: {e}")


def main():
    """Run all tests."""
    print("\n" + "🚀 STARTING VERSA CLIENT TESTS")

    # List all available models
    list_all_models()

    # Test model detection
    test_model_detection()

    # Test simple predictions (optional - requires API keys)
    print("\n" + "=" * 60)
    print("Would you like to test actual API calls? (requires valid API keys)")
    print("This will make real API calls and may incur costs.")
    user_input = input("Test API calls? (y/n): ").strip().lower()

    if user_input == 'y':
        test_simple_predictions()
    else:
        print("Skipping API call tests.")

    print("\n✅ All tests completed!")


if __name__ == "__main__":
    main()