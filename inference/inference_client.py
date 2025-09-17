import os
import sys
import base64
import time
from io import BytesIO
import gc
from typing import List, Dict, Optional, Any

import requests
import torch
from PIL import Image
import openai

try:
    from .infclient import run_inference as local_run_inference, LocalFetcher
    INFCLIENT_AVAILABLE = True
except ImportError as e:
    print(f"Warning: Could not import from '.infclient'. Local models may not work. Error: {e}", file=sys.stderr)
    LocalFetcher = None
    local_run_inference = None
    INFCLIENT_AVAILABLE = False

try:
    from utils.utils import encode as encode_pil_to_b64_str
    from utils.utils import decode as decode_b64_str_to_pil
except ImportError:
    print("FATAL: Could not import 'encode' or 'decode' from 'utils.utils'. These are required.", file=sys.stderr)
    sys.exit(1)

class InferenceClient:
    """
    Unified inference client for various visual question answering models and backends.
    Handles image encoding, message formatting, and API calls for local models,
    OpenAI (using a custom-like format as per user request), and OpenRouter.

    Few-shot examples should be provided as a list of turns, where each turn is a dictionary:
    For user: {'role': 'user', 'text': Optional[str], 'images_pil': Optional[List[PIL.Image.Image]]}
    For assistant: {'role': 'assistant', 'answer_text': str}
    """
    def __init__(self,
                 backend_name: str,
                 model_identifier: str,
                 use_local_flag: bool,
                 local_model_fs_path: Optional[str],
                 few_shot_turns: Optional[List[Dict[str, Any]]] = None):

        self.backend = backend_name.lower()
        self.model_id = model_identifier
        self.use_local = use_local_flag
        self.local_model_path = local_model_fs_path
        self.few_shot_turns = few_shot_turns if few_shot_turns else []

        self.openai_client: Any = None 
        self._or_headers: Optional[Dict[str, str]] = None
        self._or_endpoint: str = "https://openrouter.ai/api/v1/chat/completions"
        self.hf_model: Any = None
        self.hf_processor: Any = None
        self.hf_device: Any = None

        if self.use_local:
            if not INFCLIENT_AVAILABLE or not LocalFetcher or not local_run_inference:
                raise RuntimeError("LocalFetcher/run_inference from infclient.py not available. Cannot use local models.")
            fetcher = LocalFetcher(model_path=self.local_model_path, model_name=self.model_id)
            self.hf_model, self.hf_processor, self.hf_device = fetcher.get_model_params()
        elif self.backend == "openai":
            key =  os.getenv("OPENAI_LAB_KEY") or os.getenv("OPENAI_PERSONAL_KEY")
            if not key:
                raise RuntimeError("OpenAI API key not set (OPENAI_API_KEY, OPENAI_LAB_KEY, or OPENAI_PERSONAL_KEY).")
            self.openai_client = openai.OpenAI(api_key=key, timeout=360)
        elif self.backend == "openrouter":
            key = os.getenv("OPENROUTER_API_KEY") or os.getenv("OPENROUTER_LAB_TOK")
            if not key:
                raise RuntimeError("OpenRouter API key not set (OPENROUTER_API_KEY or OPENROUTER_LAB_TOK).")
            self._or_headers = {"Authorization": f"Bearer {key}", "Content-Type": "application/json"}
        else:
            raise ValueError(f"Unsupported backend in InferenceClient: {self.backend}")

    @staticmethod
    def _format_image_data_url(b64_str: str) -> str:
        return f"data:image/png;base64,{b64_str}"

    def _prepare_openai_custom_input_payload(self,
                                           prompt_text: str,
                                           target_images_pil: List[Image.Image],
                                           use_few_shot: bool) -> List[Dict[str, Any]]:
        """Prepares the 'input' payload for the user-specified OpenAI client.responses.create() format."""
        input_payload: List[Dict[str, Any]] = []

        if use_few_shot and self.few_shot_turns:
            for turn in self.few_shot_turns:
                role = turn["role"]
                content_items = []
                
                text_for_turn = turn.get("text") if role == "user" else turn.get("answer_text")
                if text_for_turn:
                    content_items.append({"type": "input_text" if role == "user" else "output_text", "text": text_for_turn})
                
                if role == "user" and turn.get("images_pil"):
                    for img_pil in turn["images_pil"]:
                        b64_str = encode_pil_to_b64_str(img_pil)
                        content_items.append({"type": "input_image", "image_url": self._format_image_data_url(b64_str)})
                
                if content_items: # Ensure content is not empty
                    input_payload.append({"role": role, "content": content_items})
        
        # Current query
        final_query_content = [{"type": "input_text", "text": prompt_text}]
        for img_pil in target_images_pil:
            b64_str = encode_pil_to_b64_str(img_pil)
            final_query_content.append({"type": "input_image", "image_url": self._format_image_data_url(b64_str)})
        input_payload.append({"role": "user", "content": final_query_content})
        
        return input_payload

    def _call_openai_custom_api(self, input_payload: List[Dict[str, Any]], max_tokens: int) -> str:
        """Calls the OpenAI API using the user-specified client.responses.create() method with retry logic."""
        if not self.openai_client:
            return "{error_openai_client_not_initialized}"

        for attempt in range(2):
            try:
                response = self.openai_client.responses.create( # type: ignore
                    model=self.model_id,
                    reasoning={"effort": "high"},
                    input=input_payload
                )

                if hasattr(response, 'output_text'):
                    return response.output_text or "{err_openai_empty_output_text}"

                if hasattr(response, 'choices') and response.choices and \
                   hasattr(response.choices[0], 'message') and response.choices[0].message and \
                   hasattr(response.choices[0].message, 'content'):
                    return response.choices[0].message.content or "{err_openai_empty_content}"

                print(f"[OpenAI Warning ({self.model_id})] Unexpected response structure: {response}", file=sys.stderr)
                return "{err_openai_unexpected_response_structure}"

            except AttributeError as ae:
                print(f"[OpenAI Error ({self.model_id})] 'responses.create' method not found on client or response structure incorrect. Error: {ae}", file=sys.stderr)
                return "{err_openai_attribute_error}"
            except Exception as e:
                print(f"[OpenAI API Error ({self.model_id})] Attempt {attempt + 1}/2 failed: {e}", file=sys.stderr)
                if hasattr(e, 'response') and hasattr(e.response, 'text'):
                     print(f"    Response body: {e.response.text}", file=sys.stderr)

                if attempt == 0:
                    print(f"[OpenAI Retry ({self.model_id})] Waiting 60 seconds before retry...", file=sys.stderr)
                    time.sleep(60)
                else:
                    return "{err_openai_api_exception}"

        return "{err_openai_api_exception}"

    def _prepare_openrouter_messages(self,
                                   prompt_text: str,
                                   target_images_pil: List[Image.Image],
                                   use_few_shot: bool) -> List[Dict[str, Any]]:
        """Prepares messages for OpenRouter, typically in standard chat completion format."""
        messages: List[Dict[str, Any]] = []

        if use_few_shot and self.few_shot_turns:
            for turn in self.few_shot_turns:
                role = turn["role"]
                content_parts: List[Dict[str, Any]] = []
                
                text_for_turn = turn.get("text") if role == "user" else turn.get("answer_text")
                if text_for_turn:
                    content_parts.append({"type": "text", "text": text_for_turn})

                if role == "user" and turn.get("images_pil"):
                    for img_pil in turn["images_pil"]:
                        b64_str = encode_pil_to_b64_str(img_pil)
                        content_parts.append({"type": "image_url", "image_url": {"url": self._format_image_data_url(b64_str)}})
                
                if content_parts:
                    if role == "assistant" and len(content_parts) == 1 and content_parts[0]["type"] == "text":
                        messages.append({"role": role, "content": content_parts[0]["text"]})
                    else:
                        messages.append({"role": role, "content": content_parts})
        
        final_query_content_parts: List[Dict[str, Any]] = [{"type": "text", "text": prompt_text}]
        for img_pil in target_images_pil:
            b64_str = encode_pil_to_b64_str(img_pil)
            final_query_content_parts.append({"type": "image_url", "image_url": {"url": self._format_image_data_url(b64_str)}})
        messages.append({"role": "user", "content": final_query_content_parts})
        
        return messages

    def _call_openrouter_api(self, messages_payload: List[Dict[str, Any]], max_tokens: int) -> str:
        """Calls the OpenRouter API with retry logic."""
        if not self._or_headers:
            return "{error_or_config_missing}"
        payload = {
            "model": self.model_id,
            "messages": messages_payload,
            "temperature": 0.001,
            "reasoning": {"effort": "high"}
        }

        for attempt in range(2):
            try:
                response = requests.post(self._or_endpoint, headers=self._or_headers, json=payload, timeout=360)
                response.raise_for_status()
                data = response.json()
                #breakpoint()

                content = data.get("choices", [{}])[0].get("message", {}).get("content", "")
                if not content and data.get("choices", [{}])[0].get("message", {}).get("tool_calls"):
                    print(f"[OpenRouter Note ({self.model_id})] Model returned tool calls instead of content.", file=sys.stderr)
                    return "{tool_call_response}"
                return content or "{error_or_empty_response}"

            except requests.exceptions.RequestException as e:
                response_text = e.response.text if e.response else "N/A"
                print(f"[OpenRouter API Request Error ({self.model_id})] Attempt {attempt + 1}/2 failed: {e}. Response: {response_text}", file=sys.stderr)

                if attempt == 0:
                    print(f"[OpenRouter Retry ({self.model_id})] Waiting 60 seconds before retry...", file=sys.stderr)
                    time.sleep(60)
                else:
                    return "{error_or_request_failed}"

            except Exception as e:
                print(f"[OpenRouter Error ({self.model_id})] Attempt {attempt + 1}/2 failed with unexpected error: {e}", file=sys.stderr)

                if attempt == 0:
                    print(f"[OpenRouter Retry ({self.model_id})] Waiting 60 seconds before retry...", file=sys.stderr)
                    time.sleep(60)
                else:
                    return "{error_or_unexpected}"

        return "{error_or_unexpected}"

    def _prepare_local_messages(self,
                                prompt_text: str,
                                target_images_pil: List[Image.Image],
                                use_few_shot: bool) -> List[Dict[str, Any]]:
        """Prepares messages for local inference, images typically passed as PIL objects."""
        local_messages: List[Dict[str, Any]] = []

        if use_few_shot and self.few_shot_turns:
            for turn in self.few_shot_turns:
                role = turn["role"]
                content_items_local: List[Dict[str, Any]] = []

                if role == "user" and turn.get("images_pil"):
                    for img_pil in turn["images_pil"]:
                        content_items_local.append({"type": "image", "image": img_pil}) # Pass PIL
                
                text_for_turn = turn.get("text") if role == "user" else turn.get("answer_text")
                if text_for_turn:
                    content_items_local.append({"type": "text", "text": text_for_turn})

                if content_items_local:
                    local_messages.append({"role": role, "content": content_items_local})
        
        final_user_content_local: List[Dict[str, Any]] = []
        for img_pil in target_images_pil:
            final_user_content_local.append({"type": "image", "image": img_pil})
        if prompt_text:
            final_user_content_local.append({"type": "text", "text": prompt_text})
        
        if final_user_content_local:
            local_messages.append({"role": "user", "content": final_user_content_local})
        
        return local_messages

    
    def ask(self,
            prompt_text: str,
            target_images_pil: List[Image.Image],
            use_few_shot_for_this_call: bool, 
            max_tokens: int = 150,
            current_item_fs_turns: Optional[List[Dict[str, Any]]] = None
            ) -> str:
        """
        Sends a request to the configured model backend.
        Args:
            prompt_text: The main text prompt for the query.
            target_images_pil: A list of PIL.Image objects for the current query.
            use_few_shot_for_this_call: Boolean indicating whether to apply few-shot prompting for this call.
            max_tokens: Maximum number of NEW tokens to generate.
            current_item_fs_turns: Optional pre-formatted few-shot turns for this specific item,
                                   overrides client's default few_shot_turns if provided and
                                   use_few_shot_for_this_call is True.
        Returns:
            The model's response string.
        """
        
        # Determine the actual few-shot turns to use for this call
        final_fs_turns_to_use: List[Dict[str, Any]] = []
        if use_few_shot_for_this_call:
            if current_item_fs_turns is not None: 
                final_fs_turns_to_use = current_item_fs_turns
            elif self.few_shot_turns: 
                final_fs_turns_to_use = self.few_shot_turns
        

        original_client_fs_turns = self.few_shot_turns
        self.few_shot_turns = final_fs_turns_to_use
        
        actually_apply_fs_content_in_payload = use_few_shot_for_this_call and bool(final_fs_turns_to_use)

        response_text = "{err_unsupported_backend_in_ask}"

        if self.use_local:
            if not local_run_inference or not self.hf_model:
                response_text = "{err_local_client_not_ready}"
            else:
                local_messages = self._prepare_local_messages(prompt_text, target_images_pil, actually_apply_fs_content_in_payload)
                if not local_messages or not any(msg.get("role") == "user" for msg in local_messages):
                     response_text = "{err_local_empty_query}"
                else:
                    response_text = local_run_inference(self.hf_model, self.hf_processor, self.model_id,
                                               messages=local_messages, temperature=0.001, max_new_tokens=max_tokens)
        
        elif self.backend == "openai":
            input_payload = self._prepare_openai_custom_input_payload(prompt_text, target_images_pil, actually_apply_fs_content_in_payload)
            response_text = self._call_openai_custom_api(input_payload, max_tokens)
        
        elif self.backend == "openrouter":
            or_messages = self._prepare_openrouter_messages(prompt_text, target_images_pil, actually_apply_fs_content_in_payload)
            response_text = self._call_openrouter_api(or_messages, max_tokens=max_tokens)

        self.few_shot_turns = original_client_fs_turns
        
        return response_text
    def cleanup_local_model(self):
        """Releases resources used by a local model."""
        if self.use_local and self.hf_model is not None:
            print(f"  Attempting to release local model: {self.model_id}")
            model_to_release = self.hf_model
            if hasattr(model_to_release, 'to') and isinstance(model_to_release, torch.nn.Module):
                try:
                    model_to_release.to("cpu")
                except Exception as e:
                    print(f"    Warning: Failed to move model to CPU: {e}", file=sys.stderr)
            
            del self.hf_model, self.hf_processor, self.hf_device
            self.hf_model = self.hf_processor = self.hf_device = None
            
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            print(f"    Local model {self.model_id} resources released attempt complete.")