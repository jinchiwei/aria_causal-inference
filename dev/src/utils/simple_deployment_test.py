#!/usr/bin/env python3
"""
Simple test to check if we can list deployments
"""

import os
import requests
from dotenv import load_dotenv
from pathlib import Path

# Load environment variables
load_dotenv(dotenv_path=Path.home() / "versa_key.env")
API_KEY = os.getenv("VERSA_API_KEY")
AZURE_ENDPOINT = os.getenv("VERSA_ENDPOINT")
API_VERSION = "2024-10-21"

def simple_list_deployments():
    """Simple function to list deployments without heavy dependencies"""
    try:
        url = f"{AZURE_ENDPOINT}/openai/deployments?api-version={API_VERSION}"
        
        headers = {
            "api-key": API_KEY,
            "Content-Type": "application/json"
        }
        
        response = requests.get(url, headers=headers)
        
        if response.status_code == 200:
            deployments = response.json()
            print("Available deployments:")
            print("-" * 50)
            
            if 'data' in deployments:
                for deployment in deployments['data']:
                    print(f"Deployment ID: {deployment.get('id', 'N/A')}")
                    print(f"Model: {deployment.get('model', 'N/A')}")
                    print(f"Status: {deployment.get('status', 'N/A')}")
                    if 'created_at' in deployment:
                        print(f"Created: {deployment.get('created_at', 'N/A')}")
                    print("-" * 30)
                    
                return [dep.get('id') for dep in deployments['data']]
            else:
                print("No deployments found or unexpected response format")
                print("Raw response:", deployments)
                return []
        else:
            print(f"Error: {response.status_code}")
            print(f"Response: {response.text}")
            return []
            
    except Exception as e:
        print(f"Error listing deployments: {e}")
        return []

if __name__ == "__main__":
    print("Checking your Azure OpenAI deployments...")
    deployments = simple_list_deployments()
    
    if deployments:
        print(f"\nFound {len(deployments)} deployments:")
        for dep in deployments:
            print(f"  - {dep}")
    else:
        print("\nNo deployments found or error occurred")
        
    print(f"\nCurrent deployment in use: gpt-4o-2024-08-06")
    print(f"API Endpoint: {AZURE_ENDPOINT}")
    print(f"API Version: {API_VERSION}")