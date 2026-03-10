#!/usr/bin/env python3
"""
Test script to check available Azure OpenAI deployments
"""

from versa_api import list_deployments, test_deployment, get_available_deployments

def main():
    print("=== Checking Available Deployments ===")
    
    # Method 1: List all deployments with details
    deployments = list_deployments()
    
    print("\n=== Testing Individual Deployments ===")
    
    # Method 2: Get list of available deployment names
    available_deployments = get_available_deployments()
    
    if available_deployments:
        print(f"Found {len(available_deployments)} available deployments")
        
        # Test each deployment
        for deployment in available_deployments:
            print(f"\nTesting deployment: {deployment}")
            test_deployment(deployment)
    else:
        print("No available deployments found")
        
    # Method 3: Test specific deployments you know about
    print("\n=== Testing Known Deployments ===")
    known_deployments = [
        "gpt-4o-2024-08-06",  # Your current deployment
        "gpt-4o-2024-05-13",  # From your commented code
        "gpt-35-turbo",       # Common deployment name
        "anthropic.claude-3-5-sonnet-20241022-v2:0"
    ]
    
    for deployment in known_deployments:
        print(f"\nTesting known deployment: {deployment}")
        test_deployment(deployment)

if __name__ == "__main__":
    main()