"""Remote LLM API clients (Gemini via REST, OpenAI via the official SDK).

`LanguageModelClient(model_name, api_key)` dispatches on the model name ("gemini..." or
"gpt.../openai...") and exposes one `prompt()` for both a single string and a batch of prompts,
with retries/backoff and bounded concurrency.
"""

import asyncio
import json
import time
from typing import Any, Dict, List, Optional, Union

import backoff
import httpx
import nest_asyncio
import openai
import requests
from tqdm import tqdm

MAX_RETRIES = 5
MAX_GEMINI_CONCURRENCY = 40
MAX_OPENAI_CONCURRENCY = 140

class LanguageModelClient:
    def __init__(self, model_name: str, api_key: str, params = {}):
        self.api_key = api_key
        self.model_name = model_name.lower()
        
        self.defaults = {}
        self.defaults["temperature"] = params["temperature"] if "temperature" in params else None
        self.defaults["max_output_tokens"] = params["max_output_tokens"] if "max_output_tokens" in params else None

        if "gemini" in self.model_name:
            self.model_type = "gemini"
            self.model_endpoint = model_name
            self.api_base_url = "https://generativelanguage.googleapis.com/v1beta/models/"
        elif "gpt" in self.model_name or "openai" in self.model_name:
            self.model_type = "openai"
            self.client = openai.OpenAI(api_key=self.api_key)
            self.model_endpoint = model_name
            self.api_base_url = "https://api.openai.com/v1/responses"

        else:
            raise ValueError(f"Unsupported model name: {model_name}. Must contain 'gemini' or 'gpt'.")
            
        print(f"Client initialized for {self.model_type} model: {self.model_endpoint}")


    def prompt(
        self,
        prompts: Union[str, List[str]],
        temperature: Optional[float] = None,
        max_output_tokens: Optional[int] = None,
        # Other Control Parameters
        system_instruction: Optional[str] = None,
        web_search: Optional[bool] = False,
    ) -> Dict[str, Any]:
        """
        Dispatches the prompt to the appropriate model implementation.
        """
        effective_config = {
            "temperature": temperature,
            "max_output_tokens": max_output_tokens,
            "web_search": web_search
        }
        
        if self.model_type == "gemini":

            if type(prompts) == str:
                return self._prompt_gemini(
                    prompt=prompts,
                    config=effective_config,
                    system_instruction=system_instruction,
                )['text']

            prompt_list = [prompts] if isinstance(prompts, str) else prompts

            coro = self._prompt_gemini_batch(
                prompt_list, 
                config=effective_config, 
                system_instruction=system_instruction,
            )

            try:
                loop = asyncio.get_running_loop()
                nest_asyncio.apply()
                return [i['text'] for i in loop.run_until_complete(coro)]
            except RuntimeError:
                return [i['text'] for i in asyncio.run(coro)]
                # return asyncio.run(coro)

            
        elif self.model_type == "openai":
            return self.batch_prompt_openai(
                prompts=prompts,
                config=effective_config,
                system_instruction=system_instruction
            )
        else:
            return {"text": f"Error: Unsupported model type {self.model_type}", "sources": []}


    def _prompt_gemini(
        self, 
        prompt: str, 
        config: Dict[str, Any], 
        system_instruction: Optional[str]
    ) -> Dict[str, Any]:
        """
        Core logic for calling the Gemini API with exponential backoff.
        """

        # --- 1. Construct the Payload ---
        # Map Python-style keys back to camelCase for the API
        generation_config = {
            "temperature": config["temperature"],
            "maxOutputTokens": config["max_output_tokens"],
            "topP": config.get("top_p", None),
            "topK": config.get("top_K", None),
        }

        payload = {
            "contents": [{"parts": [{"text": prompt}]}],
            "generationConfig": generation_config
        }

        if system_instruction:
            payload["systemInstruction"] = {"parts": [{"text": system_instruction}]}

        if config['web_search']:
            payload["tools"] = [{"google_search": {}}]

        headers = {
            "Content-Type": "application/json",
        }
        
        # Construct the full API URL
        full_url = f"{self.api_base_url}{self.model_endpoint}:generateContent?key={self.api_key}"

        # print(f"Sending request to Gemini API (Grounding: {use_google_search}, Temp: {config['temperature']})...")
        
        # --- 2. Request Loop with Exponential Backoff ---
        for attempt in range(MAX_RETRIES):
            try:
                response = requests.post(full_url, headers=headers, data=json.dumps(payload))
                response.raise_for_status()  # Raises HTTPError for bad responses (4xx or 5xx)

                # --- 3. Process Successful Response ---
                result = response.json()
                candidate = result.get('candidates', [{}])[0]
                
                # Extract generated text
                generated_text = candidate.get('content', {}).get('parts', [{}])[0].get('text', '')
                
                # Extract grounding sources if available
                sources: List[Dict[str, Optional[str]]] = []
                grounding_metadata = candidate.get('groundingMetadata')
                if grounding_metadata and grounding_metadata.get('groundingAttributions'):
                    sources = [
                        {
                            'uri': attr.get('web', {}).get('uri'),
                            'title': attr.get('web', {}).get('title'),
                        }
                        for attr in grounding_metadata['groundingAttributions']
                        if attr.get('web', {}).get('uri') and attr.get('web', {}).get('title')
                    ]
                
                return {
                    "text": generated_text,
                    "sources": sources
                }

            except requests.exceptions.HTTPError as e:
                if response.status_code in [429, 503] and attempt < MAX_RETRIES - 1:
                    delay = 2 ** attempt
                    # print(f"Attempt {attempt + 1} failed (HTTP {response.status_code}). Retrying in {delay}s...")
                    time.sleep(delay)
                else:
                    print(f"Fatal HTTP Error: {e}")
                    print(f"Response Body: {response.text}")
                    return {"text": f"Error: Failed to get response from API (Status {response.status_code}).", "sources": []}

            except requests.exceptions.RequestException as e:
                print(f"Request Error: {e}")
                return {"text": "Error: Network or connection issue.", "sources": []}
                
        return {"text": "Error: Max retries exceeded.", "sources": []}

    @backoff.on_exception(backoff.expo, openai.RateLimitError)
    def _chat_completion_with_backoff(self, client, **kwargs):
        return client.chat.completions.create(**kwargs)
    
    def batch_prompt_openai(
        self,
        prompts: Union[str, List[str]],
        config: Dict[str, Any],
        system_instruction: Optional[str] = None
    ) -> List[str]:

        if isinstance(prompts, str):
            prompts = [prompts]

        args = {}
        if self.defaults['temperature'] is not None:
            args["temperature"] = self.defaults['temperature']
        if self.defaults['max_output_tokens'] is not None:
            args["max_tokens"] = self.defaults['max_output_tokens']
        if config['temperature'] != None:
            args['temperature'] = config['temperature']
        if config['max_output_tokens'] != None:
            args['max_tokens'] = config['max_output_tokens']
        # print(args)     
        results = []
        # Process prompts in chunks to respect max concurrency
        for i in tqdm(range(0, len(prompts), MAX_OPENAI_CONCURRENCY)):
            chunk = prompts[i:i + MAX_OPENAI_CONCURRENCY]

            for prompt in chunk:
                messages = [{"role": "user", "content": prompt}]
                if system_instruction:
                    messages.insert(0, {"role": "system", "content": system_instruction})

                response = self._chat_completion_with_backoff(
                    self.client,
                    model=self.model_endpoint,
                    messages=messages,
                    **args
                )

                try:
                    text = response.choices[0].message.content.strip()
                except Exception:
                    text = str(response)

                results.append(text)

        return results

    async def _prompt_gemini_batch(
        self,
        prompts: List[str],
        config: Dict[str, Any],
        system_instruction: Optional[str],
    ) -> List[Dict[str, any]]:
        


        async def call_gemini(prompt: str) -> Dict[str, any]:
            
            generation_config = {}
            temp = config.get("temperature")
            max_tokens = config.get("max_output_tokens")

            if self.defaults['temperature'] is not None:
                generation_config["temperature"] = self.defaults['temperature']
            if self.defaults['max_output_tokens'] is not None:
                generation_config["maxOutputTokens"] = self.defaults['max_output_tokens']
            if temp is not None:
                generation_config["temperature"] = temp
            if max_tokens is not None:
                generation_config["maxOutputTokens"] = int(max_tokens)

            payload = {
                "contents": [{"parts": [{"text": prompt}]}],
            }

            if generation_config:
                payload["generationConfig"] = generation_config

            if system_instruction:
                payload["systemInstruction"] = {"parts": [{"text": system_instruction}]}

            if config.get('web_search'):
                payload["tools"] = [{"google_search": {}}]

            headers = {"Content-Type": "application/json"}
            full_url = f"{self.api_base_url}{self.model_endpoint}:generateContent?key={self.api_key}"
            # print(payload)
            for attempt in range(MAX_RETRIES):
                try:
                    async with httpx.AsyncClient() as client:
                        response = await client.post(full_url, headers=headers, json=payload, timeout=30)
                        response.raise_for_status()

                        result = response.json()
                        candidate = result.get('candidates', [{}])[0]

                        generated_text = candidate.get('content', {}).get('parts', [{}])[0].get('text', '')

                        sources: List[Dict[str, Optional[str]]] = []
                        grounding_metadata = candidate.get('groundingMetadata')
                        if grounding_metadata and grounding_metadata.get('groundingAttributions'):
                            sources = [
                                {
                                    'uri': attr.get('web', {}).get('uri'),
                                    'title': attr.get('web', {}).get('title'),
                                }
                                for attr in grounding_metadata['groundingAttributions']
                                if attr.get('web', {}).get('uri') and attr.get('web', {}).get('title')
                            ]
                        return {"text": generated_text, "sources": sources}

                except httpx.HTTPStatusError as e:
                    if response.status_code in [429, 503] and attempt < MAX_RETRIES - 1:
                        await asyncio.sleep(2 ** attempt)
                    else:
                        return {"text": f"Error: Failed (HTTP {response.status_code})", "sources": []}
                except httpx.RequestError:
                    return {"text": "Error: Network issue", "sources": []}

            return {"text": "Error: Max retries exceeded", "sources": []}

        semaphore = asyncio.Semaphore(MAX_GEMINI_CONCURRENCY)

        async def call_gemini_semaphore(prompt: str) -> Dict[str, any]:
            async with semaphore:
                return await call_gemini(prompt)  # call_gemini is your existing inner function

        # Run all prompts concurrently, respecting the max concurrency
        results = await asyncio.gather(*(call_gemini_semaphore(p) for p in prompts))
        return results
