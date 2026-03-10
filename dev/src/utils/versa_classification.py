import os
import json
import pandas as pd
import requests
import time
import base64
import urllib.parse
import re
from pathlib import Path
from typing import List, Dict, Any, Tuple

# Configuration from your original code
RETRY_SECS = 10
MAX_RETRIES = 4

# Add your environment variables here
API_KEY = os.getenv('API_KEY', '')  # Set your API key here
API_VERSION = '2024-06-01'  # For the most recent production release: https://learn.microsoft.com/en-us/azure/ai-services/openai/api-version-deprecation#latest-ga-api-release
RESOURCE_ENDPOINT = 'https://unified-api.ucsf.edu/general'  # no trailing slash--this is used by libraries as a partial URL

chat_deployments = [
    #'gpt-35-turbo',
    'gpt-4-turbo-128k',
    #'gpt-4o-mini-2024-07-18',
    #'gpt-4o-2024-11-20'
]

error_msg = "\nProvided your configuration parameters (API_KEY, API_VERSION, RESOURCE_ENDPOINT, deployment name) are valid, the majority of errors you may encounter with this code are attributable to temporary issues such as Azure server outages or other users who have triggered shared API rate limits for a given deployment. Please try again in a few minutes.\n"

def find_json_files(root_folder: str) -> List[str]:
    """
    Recursively find all JSON files in the given folder and its subfolders.
    
    Args:
        root_folder (str): Path to the root folder to search
        
    Returns:
        List[str]: List of full paths to JSON files
    """
    json_files = []
    root_path = Path(root_folder)
    
    if not root_path.exists():
        print(f"Warning: Folder {root_folder} does not exist")
        return json_files
    
    for file_path in root_path.rglob("*.json"):
        json_files.append(str(file_path))
    
    print(f"Found {len(json_files)} JSON files in {root_folder}")
    return json_files

def load_json_file(file_path: str) -> Tuple[Dict[Any, Any], str]:
    """
    Load and parse a JSON file.
    
    Args:
        file_path (str): Path to the JSON file
        
    Returns:
        Tuple[Dict[Any, Any], str]: (parsed_json_data, error_message)
    """
    try:
        with open(file_path, 'r', encoding='utf-8') as file:
            data = json.load(file)
        return data, ""
    except json.JSONDecodeError as e:
        return {}, f"JSON decode error: {str(e)}"
    except FileNotFoundError:
        return {}, f"File not found: {file_path}"
    except Exception as e:
        return {}, f"Error loading file: {str(e)}"

def create_prompt(json_data: Dict[Any, Any]) -> str:
    """
    Create the classification prompt with the JSON data embedded.
    
    Args:
        json_data (Dict[Any, Any]): The JSON data to embed in the prompt
        
    Returns:
        str: The complete prompt with embedded JSON data
    """
    json_str = json.dumps(json_data, indent=2)
    
    prompt = f"""Classify the given MRI sequence data into one of the following categories: T1, T2, T1 post contrast, FLAIR, ADC, DWI, GRE, or other. The classification should be based on ALL the dicom header information, series description, the echo time (TE) and repetition time (TR) of the sequence. T1 sequences typically have a TE between 10-20 ms and a TR between 500-800 ms. T2 sequences have a TE between 80-120 ms and a TR between 2000-5000 ms. T1 post contrast sequences are similar to T1 sequences, but the data will indicate the use of a gadolinium contrast agent. FLAIR sequences have a TE between 100-150 ms and a TR between 6000-10000 ms. ADC sequences have a TE between 60-90 ms and a TR between 3000-5000 ms. DWI sequences have a TE between 50-100 ms and a TR between 3000-6000 ms. GRE sequences have a TE between 15-30 ms and a TR between 500-1000 ms. Any sequences such as reformatted or localizers or ones that do not fit these parameters should be classified as "other". Provide the classification for the given MRI sequence data.

```json
{json_str}
```

Use the following json format to answer:
{{
    "answer": "<your choice from the list>",
    "reasoning": "<your reasoning as to why you chose the option>"
}}"""
    
    return prompt

def post_request(url: str, headers: Dict[str, str], body: str) -> requests.Response:
    """
    Make a POST request with error handling.
    
    Args:
        url (str): The URL to make the request to
        headers (Dict[str, str]): Request headers
        body (str): Request body
        
    Returns:
        requests.Response: The response object
    """
    response = requests.post(url, headers=headers, data=body)
    response.raise_for_status()
    return response

def exception_handler(retries: int, deployment_id: str, e: Exception) -> int:
    """
    Handle exceptions with retry logic.
    
    Args:
        retries (int): Current retry count
        deployment_id (str): The deployment being tested
        e (Exception): The exception that occurred
        
    Returns:
        int: Updated retry count
        
    Raises:
        Exception: If max retries exceeded
    """
    if retries >= MAX_RETRIES:
        print(f'Failed attempt {retries+1} of {MAX_RETRIES+1}.')
        print(error_msg)
        raise Exception(f"Test failed for deployment: {deployment_id}, Error received: {e}")
    else:
        print(f'Failed attempt {retries+1} of {MAX_RETRIES + 1}. Waiting {RETRY_SECS} secs before next attempt...')
        
    retries += 1
    time.sleep(RETRY_SECS)
    return retries

def call_openai_api(deployment_id: str, prompt: str) -> Dict[str, Any]:
    """
    Call OpenAI API with the given deployment and prompt.
    
    Args:
        deployment_id (str): The deployment to use
        prompt (str): The prompt to send
        
    Returns:
        Dict[str, Any]: The API response data
    """
    url = f'{RESOURCE_ENDPOINT}/openai/deployments/{deployment_id}/chat/completions?api-version={API_VERSION}'
    
    body = json.dumps({
        "messages": [{"role": "user", "content": prompt}],
        "temperature": 0.1,  # Lower temperature for more consistent results
        "max_tokens": 500
    })
    
    headers = {
        'Content-Type': 'application/json',
        'api-key': API_KEY
    }
    
    retries = 0
    
    while True:
        try:
            response = post_request(url, headers, body)
            response_data = json.loads(response.text)
            return response_data
        except Exception as e:
            retries = exception_handler(retries, deployment_id, e)

def parse_model_response(response_text: str) -> Tuple[str, str]:
    """
    Parse the model response to extract answer and reasoning.
    
    Args:
        response_text (str): The raw response text from the model
        
    Returns:
        Tuple[str, str]: (answer, reasoning)
    """
    try:
        # Try to find JSON in the response
        json_start = response_text.find('{')
        json_end = response_text.rfind('}') + 1
        
        if json_start != -1 and json_end != -1:
            json_str = response_text[json_start:json_end]
            parsed_response = json.loads(json_str)
            
            answer = parsed_response.get('answer', 'N/A')
            reasoning = parsed_response.get('reasoning', 'N/A')
            
            return answer, reasoning
        else:
            # If no JSON found, return the whole response as reasoning
            return 'N/A', response_text
            
    except json.JSONDecodeError:
        # If JSON parsing fails, return the whole response as reasoning
        return 'N/A', response_text
    except Exception as e:
        return 'Error', f'Error parsing response: {str(e)}'

def process_json_files(root_folder: str, output_folder: str = "results") -> None:
    """
    Process all JSON files in the folder and generate results for each model.
    
    Args:
        root_folder (str): Path to the root folder containing JSON files
        output_folder (str): Path to the output folder for CSV results
    """
    # Create output folder if it doesn't exist
    Path(output_folder).mkdir(exist_ok=True)
    
    # Find all JSON files
    json_files = find_json_files(root_folder)
    
    if not json_files:
        print("No JSON files found!")
        return
    
    # Process each deployment model
    for deployment_id in chat_deployments:
        print(f"\nProcessing with deployment: {deployment_id}")
        
        results = []
        
        for file_path in json_files:
            print(f"Processing: {file_path}")
            
            # Load JSON file
            json_data, error = load_json_file(file_path)
            
            if error:
                print(f"Error loading {file_path}: {error}")
                results.append({
                    'file_path': file_path,
                    'answer': 'Error',
                    'reasoning': error
                })
                continue
            
            # Create prompt
            prompt = create_prompt(json_data)
            
            try:
                # Call API
                response_data = call_openai_api(deployment_id, prompt)
                
                # Extract response content
                response_text = response_data.get('choices', [{}])[0].get('message', {}).get('content', '')
                
                # Parse response
                answer, reasoning = parse_model_response(response_text)
                
                results.append({
                    'file_path': file_path,
                    'answer': answer,
                    'reasoning': reasoning
                })
                
                print(f"  Answer: {answer}")
                
            except Exception as e:
                print(f"Error processing {file_path}: {str(e)}")
                results.append({
                    'file_path': file_path,
                    'answer': 'Error',
                    'reasoning': str(e)
                })
        
        # Save results to CSV
        df = pd.DataFrame(results)
        csv_filename = f"{deployment_id}_results.csv"
        csv_path = Path(output_folder) / csv_filename
        df.to_csv(csv_path, index=False)
        
        print(f"Results saved to: {csv_path}")
        print(f"Processed {len(results)} files for {deployment_id}")

def main():
    """
    Main function to run the MRI sequence classification pipeline.
    """
    # Set your input folder path here
    input_folder =  'loremipsum' # Update this path
    output_folder = 'loremipsum' # Default output folder
    
    # Validate environment variables
    if not API_KEY or not API_VERSION or not RESOURCE_ENDPOINT:
        print("Error: Please set API_KEY, API_VERSION, and RESOURCE_ENDPOINT environment variables")
        return
    
    print("Starting MRI Sequence Classification Pipeline")
    print(f"Input folder: {input_folder}")
    print(f"Output folder: {output_folder}")
    
    # Process all JSON files
    process_json_files(input_folder, output_folder)
    
    print("\nPipeline completed!")

if __name__ == "__main__":
    main()