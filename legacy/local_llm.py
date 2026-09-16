"""Local HuggingFace LLMs (transformers): batched generation + likelihood scoring.

`LLM(model_name, ...)` loads a causal LM on GPU/CPU and exposes:
  - `prompt(prompts)`                       — batched greedy/sampled generation
  - `prompt_likelihood` / `completion_likelihood` — log-prob scoring of completions
  - `calibrated_class_probs` / `reasoning_calibrated_probs` — null-baselined classification

`HF_models` maps short aliases (e.g. 'l32-3bi', 'g3-4i') to hub checkpoint names.
"""

import gc
from threading import Lock

import torch
from huggingface_hub import login
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer

from src.utils.llm_calls.hf_models import HF_models

_LOCK = Lock()




class LLM:
    def __init__(
        self,
        model_name: str,
        huggingface_token: str = None,
        dtype=torch.bfloat16,      # model precision
        max_new_tokens: int = 32,  # default max tokens to generate
        temperature: float = 0.0,  # default temperature (0 -> greedy)
        do_sample: bool = False,    # default sampling strategy
        batch_size: int = 32
    ):
        if huggingface_token:
            login(huggingface_token)
        
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        print(self.device)
        if self.device == "cpu":
            print("Warning: GPU not available, using CPU which may be slow.")
        
        # Tokenizer setup
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        self.tokenizer.padding_side = "left"
        # Model setup
        self.model = AutoModelForCausalLM.from_pretrained(
            model_name,
            device_map="cuda" if self.device=="cuda" else None,
            dtype=dtype if self.device=="cuda" else torch.float32,
            low_cpu_mem_usage=True
        )

        # Default generation parameters
        self.max_new_tokens = max_new_tokens
        self.temperature = temperature
        self.do_sample = do_sample
        self.batch_size = batch_size

        self.model.eval()


    def prompt(self, prompts: list[str], **kwargs):

        max_new_tokens = kwargs.pop("max_new_tokens", self.max_new_tokens)
        temperature = kwargs.pop("temperature", self.temperature)
        do_sample = kwargs.pop("do_sample", self.do_sample)
        if not do_sample:
            generation_args = dict(
                max_new_tokens=max_new_tokens,
                do_sample=do_sample,
                pad_token_id=self.tokenizer.pad_token_id,
                **kwargs
            )
        else:
            generation_args = dict(
                max_new_tokens=max_new_tokens,
                temperature=temperature,
                do_sample=do_sample,
                pad_token_id=self.tokenizer.pad_token_id,
                **kwargs
            )
        
        results = []

        for i in tqdm(range(0, len(prompts), self.batch_size)):
            batch = prompts[i:i + self.batch_size]

            inputs = self.tokenizer(
                batch,
                return_tensors="pt",
                padding=True,
                truncation=False
            ).to(self.device)

            with torch.no_grad():
                outputs = self.model.generate(**inputs, **generation_args)

            for j, o in enumerate(outputs):
                gen = o[inputs['input_ids'].shape[1]:]
                results.append(self.tokenizer.decode(gen, skip_special_tokens=True))

            del inputs
            del outputs
            torch.cuda.empty_cache()
            torch.cuda.ipc_collect()

        return results

    def old_prompt_likelihood(self, prompts: list[str], max_new_tokens: int = None, device: str = None):
        if device is None:
            device = self.device
        if max_new_tokens is None:
            max_new_tokens = self.max_new_tokens

        # Ensure left-padding for causal LM
        self.tokenizer.padding_side = "left"

        # Step 1: generate outputs
        self.model.eval()
        inputs = self.tokenizer(
            prompts,
            return_tensors="pt",
            padding=True,
            truncation=False
        ).to(device)

        with torch.no_grad():
            generated = self.model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                pad_token_id=self.tokenizer.pad_token_id
            )

        outputs = []
        completions = []

        for i, seq in enumerate(generated):
            # slice off prompt tokens
            gen_tokens = seq[inputs['input_ids'].shape[1]:]
            outputs.append(self.tokenizer.decode(gen_tokens, skip_special_tokens=True))
            completions.append(outputs[-1])

        # Step 2: compute log-probs using method 2
        full_prompts = [p + c for p, c in zip(prompts, completions)]
        with _LOCK:
            full_inputs = self.tokenizer(
                full_prompts,
                #max_length=1024,
                truncation=False,
                padding=True,
                return_tensors="pt",
                add_special_tokens=False
            )
        input_ids = full_inputs.input_ids.to(device)
        attention_mask = full_inputs.attention_mask.to(device)

        with torch.no_grad():
            model_outputs = self.model(input_ids=input_ids, attention_mask=attention_mask)
            logits = model_outputs.logits
            log_probs = torch.nn.functional.log_softmax(logits, dim=-1)

        # shift input_ids for causal LM
        shifted_input_ids = input_ids[:, 1:].unsqueeze(-1)
        chosen_log_probs = torch.gather(log_probs, dim=-1, index=shifted_input_ids).squeeze(-1).cpu()

        # compute per-completion log-probs (exclude prompt tokens)
        with _LOCK:
            completions_ids = self.tokenizer(
                completions,
                #max_length=1024,
                add_special_tokens=False,
                padding=False
            ).input_ids

        sequence_log_probs = [
            chosen_log_probs[i, -len(c):].mean().exp().item()  # geometric mean per completion
            for i, c in enumerate(completions_ids)
        ]

        # cleanup
        del input_ids, attention_mask, model_outputs, logits, chosen_log_probs
        gc.collect()
        torch.cuda.empty_cache()

        return outputs, sequence_log_probs
    
    def prompt_likelihood(self, prompts, **kwargs):
        all_outputs = self.prompt(prompts, **kwargs)
        all_scores = []

        for i in tqdm(range(len(all_outputs))):
            chosen_log_probs = self._completion_likelihood(prompts[i], [all_outputs[i]])
            lp = chosen_log_probs[0]  # token-level log-probs tensor
            avg_logprob = lp.sum().item() / len(lp)  # length-normalized
            all_scores.append(avg_logprob)

        return [all_outputs, all_scores]

    def _completion_likelihood(self, prompt: str, completions: list[str]):
        assert self.tokenizer.padding_side == "left"
        assert prompt, "Prompt must be non-empty."

        full_prompts = [prompt + c for c in completions]

        # Tokenize the full sequences
        with _LOCK:
            full_inputs = self.tokenizer(
                full_prompts,
                # max_length=1024,
                truncation=False,
                padding=True,
                return_tensors="pt",
                add_special_tokens=False,
            )

        input_ids = full_inputs.input_ids.to(self.device)
        attention_mask = full_inputs.attention_mask.to(self.device)

        with torch.no_grad():
            outputs = self.model(input_ids=input_ids, attention_mask=attention_mask)

        logits = outputs.logits
        log_probs = torch.nn.functional.log_softmax(logits, dim=-1)

        # Shift to align logits for next-token prediction
        shifted_input_ids = input_ids[:, 1:].unsqueeze(-1)
        chosen_log_probs = torch.gather(log_probs[:, :-1, :], dim=-1, index=shifted_input_ids).squeeze(-1).cpu()

        # Keep only the tokens of the completions (exclude prompt)
        with _LOCK:
            completions_ids = self.tokenizer(
                completions,
                add_special_tokens=False,
                padding=False
            ).input_ids

        chosen_log_probs = [
            chosen_log_probs[i, -len(c):] for i, c in enumerate(completions_ids)
        ]

        # Clean up memory
        del input_ids, attention_mask, outputs, logits, log_probs
        del shifted_input_ids, full_inputs, completions_ids, full_prompts
        gc.collect()
        torch.cuda.empty_cache()



        return chosen_log_probs
    
    def completion_likelihood(self, prompts: list[str], completions_per_prompt: list[list[str]]):
        assert len(prompts) == len(completions_per_prompt), \
            "Each prompt must have a matching list of completions."
        all_results = []

        for prompt, completions in zip(prompts, completions_per_prompt):
            result = self._completion_likelihood(prompt, completions)
            all_results.append(result)

        return all_results
    
    def calibrated_class_probs(self, prompts: list[str], classes: list[str],
                            multiple_choice_classes = None, null_prompts=None, temperature=1.0, null_clip=(-20, -7)):
        """
        Compute calibrated probabilities over candidate classes for multiple prompts.

        Improvements:
        1️⃣ Null baseline: uses clipped mean to reduce extreme outlier null prompts.
        2️⃣ Length-normalization: divides log-likelihood sums by token count.
        3️⃣ Softmax temperature: flattens overconfident probabilities.
        4️⃣ Multi-prompt support: outputs tensor of shape (num_prompts, num_classes).

        Args:
            prompts: list of strings to classify
            classes: list of candidate labels
            null_prompts: list of prompts used to compute null baseline
            temperature: softmax temperature (>1 flattens probabilities)
            null_clip: tuple (min, max) to clip null log-likelihoods
        """
        if null_prompts is None:
            null_prompts = ["\n", "\n\n", "?", "[NULL]"]

        # 1️⃣ Compute null baseline once (same for all prompts)
        null_avg_list = []
        for null_prompt in null_prompts:
            null_chosen_log_probs = self._completion_likelihood(null_prompt, classes)
            null_total_log_probs = torch.tensor([lp.sum().item() for lp in null_chosen_log_probs])
            null_lengths = torch.tensor([len(lp) for lp in null_chosen_log_probs])
            null_avg_list.append(null_total_log_probs / null_lengths)

        # Stack nulls and clip extreme values
        null_stack = torch.stack(null_avg_list, dim=0)
        null_stack = torch.clamp(null_stack, min=null_clip[0], max=null_clip[1])
        null_avg_log_probs = null_stack.mean(dim=0)  # clipped mean for robust baseline

        # 2️⃣ Compute calibrated probabilities for each prompt
        all_probs = []
        all_loglikelihoods = []
        for i in tqdm(range(len(prompts))):
            prompt = prompts[i]

            prompt = prompt.replace(
                "Respond with only one of {A, B, C, D} with no further explaination or words.",
                ""
            )
            prompt = prompt.replace("\n\n","\n")

            # print(prompt)
            if multiple_choice_classes != None: _classes = multiple_choice_classes[i]
            else: _classes = classes
            chosen_log_probs = self._completion_likelihood(prompt, _classes)
            total_log_probs = torch.tensor([lp.sum().item() for lp in chosen_log_probs])
            lengths = torch.tensor([len(lp) for lp in chosen_log_probs])
            avg_log_probs = total_log_probs / lengths  # length-normalized
            calibrated_scores = avg_log_probs - null_avg_log_probs  # subtract robust null baseline
            probs = torch.softmax(calibrated_scores / temperature, dim=0)  # flatten with temperature
            all_probs.append(probs)
            all_loglikelihoods.append(calibrated_scores)

        all_probs_list = [p.tolist() for p in all_probs]
        all_loglikelihoods_list = [ll.tolist() for ll in all_loglikelihoods]
        return [all_loglikelihoods_list, all_probs_list]  # shape: (num_prompts, num_classes)


    def reasoning_calibrated_probs(self, prompts: list[str], classes: list[str],
                                reasoning_max_len=40, temperature=1.0,
                                null_reasoning=None):

        all_probs = []

        for prompt in prompts:
            # 1️⃣ Generate reasoning for each class
            candidate_prompts = [
                f"{prompt} {cls}. Explain briefly your reasoning for {cls}: "
                for cls in classes
            ]
            reasoning_for_classes = self.prompt(candidate_prompts, max_new_tokens=reasoning_max_len)

            # 2️⃣ Compute log-likelihoods of class + reasoning
            completions = [f"{cls} reason: {r}" for cls, r in zip(classes, reasoning_for_classes)]
            chosen_log_probs = self._completion_likelihood(
                prompt=prompt,
                completions=completions
            )
            # Length-normalize
            log_likelihoods = torch.tensor([lp.sum().item() / len(lp) for lp in chosen_log_probs])

            # 3️⃣ Null baseline subtraction
            if null_reasoning is not None:
                null_completions = [f"{cls} reason: {null_reasoning}" for cls in classes]
                null_log_probs = self._completion_likelihood(prompt, null_completions)
                null_ll = torch.tensor([lp.sum().item() / len(lp) for lp in null_log_probs])
                log_likelihoods = log_likelihoods - null_ll

            # 4️⃣ Softmax with temperature
            probs = torch.softmax(log_likelihoods / temperature, dim=0)
            all_probs.append(probs)

        return torch.stack(all_probs, dim=0)  