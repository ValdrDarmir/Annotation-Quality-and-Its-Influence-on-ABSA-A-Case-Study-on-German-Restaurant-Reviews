import requests
import json
import time
import os
import sys

GEMMA_API = "http://132.199.138.16:11434/api/generate"
# GEMMA_API ="http://132.199.143.117:11434/api/generate"


sys.path.append(os.path.abspath(os.path.join(
    os.path.dirname(__file__), '..', '..')))
# from keys import config

# GEMMA_API_ALT = config.GEMMA_API_ALT
# GEMMA_API = config.GEMMA_API
print(GEMMA_API)

class LLM:
    def __init__(self, base_model="gemma3:27b", api_url=GEMMA_API, parameters=None):
        self.model_name = base_model
        self.api_url = api_url.rstrip("/")  # remove trailing slash
        self.parameters = parameters or []

        self.headers = {
            "Content-Type": "application/json"
        }

    def predict(self, prompt, seed=0, stop=["]"], temperature=0.8, num_ctx=4096, request_timeout=20):
        """
        Perform prediction using the GEMMA API with a timeout.

        Returns:
            response_text (str), duration (float)
        Raises:
            TimeoutError if the request times out.
            RuntimeError for other request failures.
        """
        prediction_start_time = time.time()

        payload = {
            "model": self.model_name,
            "prompt": prompt,
            "stop": stop,
            "options": {
                "temperature": temperature,
                "seed": seed,
                "num_ctx": num_ctx,
            },
            "stream": False
        }

        # Add extra parameters from initialization
        for param in self.parameters:
            payload[param["name"]] = param["value"]

        try:
            response = requests.post(
                self.api_url,
                headers=self.headers,
                data=json.dumps(payload),
                timeout=request_timeout  # added timeout!
            )
            response.raise_for_status()  # raise error for bad HTTP responses
        except requests.exceptions.Timeout:
            raise TimeoutError(f"Request to {self.api_url} timed out after {request_timeout} seconds")
        except requests.exceptions.RequestException as e:
            raise RuntimeError(f"API request failed: {e}")

        response_json = response.json()
        response_text = response_json.get("response") or response_json.get("text") or ""

        # Ensure it ends with "]" if needed
        if response_text and not response_text.endswith("]"):
            response_text += "]"

        duration = time.time() - prediction_start_time
        return response_text, duration