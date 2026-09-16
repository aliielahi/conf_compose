"""Short aliases for HuggingFace hub checkpoints (kept torch-free so it's cheap to import)."""

HF_models = {
    'l32-1bi': 'meta-llama/Llama-3.2-1B-Instruct',
    'l32-3bi': 'meta-llama/Llama-3.2-3B-Instruct',
    'l31-8bi': 'meta-llama/Meta-Llama-3.1-8B-Instruct',
    'l31-70bi': 'meta-llama/Meta-Llama-3.1-70B-Instruct',
    'l33-70bi': 'meta-llama/Llama-3.3-70B-Instruct',

    'mist-7b': 'mistralai/Mistral-7B-v0.3',
    'mist-7bi': 'mistralai/Mistral-7B-Instruct-v0.3',
    'mistm-8bi': 'mistralai/Ministral-8B-Instruct-2410',
    'mists-24bi': 'mistralai/Mistral-Small-24B-Instruct-2501',
    'mist8x7b': 'mistralai/Mixtral-8x7B-v0.1',  # 56B
    'mist8x7bi': 'mistralai/Mixtral-8x7B-Instruct-v0.1',

    'phi3mii': 'microsoft/Phi-3.5-mini-instruct',
    'phi3smi': 'microsoft/Phi-3-small-8k-instruct',
    'phi3mei': 'microsoft/Phi-3-medium-4k-instruct',  # 14B
    'phi4mii': 'microsoft/Phi-4-mini-instruct',  # 3.8B
    'phi4i': 'microsoft/phi-4',  # 14B

    'g22': 'google/gemma-2-2b',
    'g29': 'google/gemma-2-9b',

    'g2-2i': 'google/gemma-2-2b-it',
    'g2-9i': 'google/gemma-2-9b-it',
    'g2-27i': 'google/gemma-2-27b-it',

    'g3-1i': 'google/gemma-3-1b-it',
    'g3-4i': 'google/gemma-3-4b-it',
    'g3-12i': 'google/gemma-3-12b-it',
    'g3-27i': 'google/gemma-3-27b-it',

    'q25-7bi': 'Qwen/Qwen2.5-7B-Instruct',
    'q25-14bi': 'Qwen/Qwen2.5-14B-Instruct',
    'q25-32bi': 'Qwen/Qwen2.5-32B-Instruct',
    'q3-4bi': 'Qwen/Qwen3-4B-Instruct-2507',
}
