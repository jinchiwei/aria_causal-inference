import os
import csv
import datetime
import getpass
import pandas as pd
from anthropic import AnthropicBedrock
from openai import AzureOpenAI
from dotenv import load_dotenv
from pathlib import Path
from sklearn.metrics import (
    confusion_matrix,
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
)
import seaborn as sns
import matplotlib.pyplot as plt
from tqdm import tqdm
import requests

# Load environment variables from the .env file
load_dotenv(dotenv_path=Path.home() / "versa_key.env")
API_KEY = os.getenv("VERSA_API_KEY")
AZURE_ENDPOINT = os.getenv("VERSA_ENDPOINT")
API_VERSION = "2025-04-01-preview"
MODEL_ZOO_OPENAI = [
    "gpt-35-turbo",
    "gpt-4-turbo-128k",
    "gpt-4o-2024-08-06",
    "gpt-4o-mini-2024-07-18",
    "gpt-4-turbo-2024-04-09",
    "gpt-4.1-2025-04-14",
    "o1-mini-2024-09-12",
    "o1-2024-12-17",
    "o3-mini-2025-01-31",
    "o4-mini-2025-04-16",
    "gpt-5-2025-08-07",
    "gpt-5-mini-2025-08-07",
    "gpt-5-nano-2025-08-07"
]
MODEL_ZOO_ANTHROPIC = [
    "anthropic.claude-3-5-sonnet-20240620-v1:0",
    "anthropic.claude-3-5-sonnet-20241022-v2:0",
    "us.anthropic.claude-sonnet-4-20250514-v1:0",
    "us.anthropic.claude-opus-4-1-20250805-v1:0",
    "us.anthropic.claude-opus-4-5-20251101-v1:0",
    "us.anthropic.claude-haiku-4-5-20251001-v1:0"
]

DEPLOYMENT = "gpt-5-2025-08-07"
# DEPLOYMENT = "anthropic.claude-3-5-sonnet-20241022-v2:0"

# Initialize both clients
openai_client = AzureOpenAI(
    api_key=API_KEY, azure_endpoint=AZURE_ENDPOINT, api_version=API_VERSION
)

anthropic_client = AnthropicBedrock(
    aws_access_key=os.getenv("AWS_ACCESS_KEY_ID"),
    aws_secret_key=os.getenv("AWS_SECRET_ACCESS_KEY"),
    aws_region=os.getenv("AWS_REGION"),
    base_url=os.getenv("ANTHROPIC_BEDROCK_BASE_URL")
)

# Class for handling both Azure OpenAI and Anthropic Bedrock interactions
class VersaAI:
    def __init__(self, deployment=DEPLOYMENT):
        self.deployment = deployment
        # Determine which client to use based on model name
        self.client_type = self._determine_client_type(deployment)
        print(f"Initialized VersaAI with model: {deployment} (using {self.client_type} client)")

    def _determine_client_type(self, model_name: str) -> str:
        """Determine if the model is OpenAI or Anthropic based on its name."""
        # Check if model is in the Anthropic list
        if model_name in MODEL_ZOO_ANTHROPIC:
            return "anthropic"
        # Check if model is in the OpenAI list
        elif model_name in MODEL_ZOO_OPENAI:
            return "openai"
        # Check for patterns in model name
        elif "anthropic" in model_name.lower() or "claude" in model_name.lower():
            return "anthropic"
        elif "gpt" in model_name.lower() or "o1" in model_name.lower() or "o3" in model_name.lower() or "o4" in model_name.lower():
            return "openai"
        else:
            # Default to OpenAI if unsure
            print(f"Warning: Could not determine client type for model '{model_name}', defaulting to OpenAI")
            return "openai"

    def predict(self, prompt: str, verbose: bool = False) -> str:
        try:
            if self.client_type == "anthropic":
                # Use Anthropic Bedrock client
                response = anthropic_client.messages.create(
                    model=self.deployment,
                    messages=[{"role": "user", "content": prompt}],
                    max_tokens=4096  # Anthropic requires max_tokens
                )
                completion = response.content[0].text if hasattr(response.content[0], 'text') else str(response.content[0])
                # Log usage for Anthropic (if available)
                if hasattr(response, 'usage'):
                    self.log_usage(response.usage, client_type="anthropic")
            else:
                # Use Azure OpenAI client
                response = openai_client.chat.completions.create(
                    model=self.deployment,  # This must match the custom deployment name you chose for your model
                    messages=[{"role": "user", "content": prompt}],
                )
                completion = response.choices[0].message.content
                # Log usage for OpenAI
                self.log_usage(response.usage, client_type="openai")

            if verbose:
                print(f"User: {prompt}\nResponse: {completion}")
                print(f"Model: {self.deployment} (Client: {self.client_type})")

            return completion
        except Exception as e:
            print(f"Error in prediction with {self.client_type} client: {e}")
            return "Error in API response"

    @staticmethod
    def log_usage(usage, client_type="openai"):
        """Log usage statistics for both OpenAI and Anthropic models."""
        log_file = "/data/rauschecker2/jkw/aria/dev/src/utils/azure_versa_usage.csv"

        try:
            with open(log_file, "a+", newline="") as fp:
                writer = csv.writer(fp)

                if client_type == "anthropic":
                    # Anthropic usage format might be different
                    if hasattr(usage, 'input_tokens') and hasattr(usage, 'output_tokens'):
                        writer.writerow(
                            [
                                usage.input_tokens,
                                usage.output_tokens,
                                datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                                getpass.getuser(),
                                "anthropic"  # Add client type to log
                            ]
                        )
                else:
                    # OpenAI usage format
                    writer.writerow(
                        [
                            usage.prompt_tokens,
                            usage.completion_tokens,
                            datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                            getpass.getuser(),
                            "openai"  # Add client type to log
                        ]
                    )
        except Exception as e:
            print(f"Warning: Could not log usage: {e}")


def classify_report_from_df(
    df: pd.DataFrame, prompt_template: str, verbose=False
) -> pd.DataFrame:
    """
    Classifies cases based on the given radiology "Report Text" in the
    DataFrame. A new column "Prediction" will be added with the model's classification
    output.

    Args:
        df (pd.DataFrame): Input dataframe. It is expected to contain a column named
            "Report Text", which includes the radiology reports to classify.
        prompt_template (str): The template prompt to be prepended to each report for
            the classification.
        verbose (bool, optional): If True, prints the prompt and the model's response
            for each report. Defaults to False.

    Returns:
        pd.DataFrame: The input dataframe with an additional "Prediction" column
            containing the model's output.
    """
    versa_ai = VersaAI()

    def classify_report(report_text:str) -> str:
        try:
            prompt = str(prompt_template) + str(report_text)
        except Exception as e:
            print(f"Error in combining prompt and report: {e}")
            print(f"Report Text: {report_text}")
            return ""
        return versa_ai.predict(prompt, verbose=verbose)

    # Enable progress bar for pandas apply
    tqdm.pandas()

    # Apply the classification function with a progress bar
    df["Prediction"] = df["Report Text"].progress_apply(classify_report)

    return df


# Function to calculate performance metrics
def calculate_metrics(df: pd.DataFrame, valid_labels: list):
    scores = df[["Label", "Prediction"]]
    scores = scores[scores.Prediction.isin(valid_labels)].dropna()

    y_true = scores["Label"]
    y_pred = scores["Prediction"]

    conf_matrix = confusion_matrix(y_true, y_pred, labels=valid_labels)
    accuracy = accuracy_score(y_true, y_pred)
    precision = precision_score(y_true, y_pred, average="weighted")
    recall = recall_score(y_true, y_pred, average="weighted")
    f1 = f1_score(y_true, y_pred, average="weighted")

    print(f"Accuracy: {accuracy:.2f}")
    print(f"Weighted Precision: {precision:.2f}")
    print(f"Weighted Recall: {recall:.2f}")
    print(f"Weighted F1 Score: {f1:.2f}")

    ax = plot_confusion_matrix(conf_matrix, valid_labels)
    return ax


# Function to plot confusion matrix
def plot_confusion_matrix(conf_matrix, valid_labels, save_to: str = None):
    ax = sns.heatmap(
        conf_matrix,
        annot=True,
        fmt="g",
        cmap="Blues",
        cbar=False,
        xticklabels=valid_labels,
        yticklabels=valid_labels,
    )
    ax.set_xlabel("Predicted Label")
    ax.set_ylabel("True Label")
    ax.set_title(
        f"Confusion Matrix for Meningioma report classification with Versa {DEPLOYMENT}"
    )
    ax.figure.set_dpi(600)
    if save_to:
        ax.figure.savefig(save_to, dpi=600)
    return ax


def list_deployments():
    """
    List available deployments in your Azure OpenAI resource.
    Uses the Azure OpenAI REST API to get deployment information.
    """
    try:
        # Construct the API URL for listing deployments
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
                    print(f"Created: {deployment.get('created_at', 'N/A')}")
                    print("-" * 30)
            else:
                print("No deployments found or unexpected response format")
                print("Raw response:", deployments)
                
            return deployments
        else:
            print(f"Error: {response.status_code}")
            print(f"Response: {response.text}")
            return None
            
    except Exception as e:
        print(f"Error listing deployments: {e}")
        return None


def test_deployment(deployment_name: str):
    """
    Test if a specific deployment is working by making a simple API call.
    
    Args:
        deployment_name (str): Name of the deployment to test
    """
    try:
        test_client = AzureOpenAI(
            api_key=API_KEY, 
            azure_endpoint=AZURE_ENDPOINT, 
            api_version=API_VERSION
        )
        
        response = test_client.chat.completions.create(
            model=deployment_name,
            messages=[{"role": "user", "content": "Hello, can you respond with just 'OK'?"}],
            max_tokens=10
        )
        
        print(f"✅ Deployment '{deployment_name}' is working!")
        print(f"Response: {response.choices[0].message.content}")
        return True
        
    except Exception as e:
        print(f"❌ Deployment '{deployment_name}' failed: {e}")
        return False


def get_available_deployments():
    """
    Get a list of deployment names that are currently available.
    Returns a list of deployment IDs/names.
    """
    deployments_info = list_deployments()
    if deployments_info and 'data' in deployments_info:
        return [dep.get('id') for dep in deployments_info['data'] if dep.get('status') == 'succeeded']
    return []


def list_all_models():
    """
    List all available models from both OpenAI and Anthropic.
    Returns a dictionary with provider names as keys and model lists as values.
    """
    all_models = {
        "openai": MODEL_ZOO_OPENAI,
        "anthropic": MODEL_ZOO_ANTHROPIC
    }

    print("=" * 60)
    print("AVAILABLE MODELS IN VERSAAI")
    print("=" * 60)

    print("\n📊 OpenAI Models (Azure):")
    print("-" * 40)
    for model in MODEL_ZOO_OPENAI:
        print(f"  • {model}")

    print("\n🤖 Anthropic Models (Bedrock):")
    print("-" * 40)
    for model in MODEL_ZOO_ANTHROPIC:
        print(f"  • {model}")

    print("\n" + "=" * 60)
    print(f"Total: {len(MODEL_ZOO_OPENAI)} OpenAI + {len(MODEL_ZOO_ANTHROPIC)} Anthropic = {len(MODEL_ZOO_OPENAI) + len(MODEL_ZOO_ANTHROPIC)} models")
    print("=" * 60)

    return all_models


def get_model_info(model_name: str):
    """
    Get information about a specific model including which client it uses.
    """
    versa = VersaAI(model_name)
    info = {
        "model_name": model_name,
        "client_type": versa.client_type,
        "is_in_openai_list": model_name in MODEL_ZOO_OPENAI,
        "is_in_anthropic_list": model_name in MODEL_ZOO_ANTHROPIC
    }

    print(f"\nModel Information for: {model_name}")
    print("-" * 40)
    print(f"Client Type: {info['client_type']}")
    print(f"In OpenAI List: {info['is_in_openai_list']}")
    print(f"In Anthropic List: {info['is_in_anthropic_list']}")

    return info
