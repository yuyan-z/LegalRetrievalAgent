OPENAI_CONFIG = {
    "model": None,
    "base_url": None,
    "api_key": None,

    # Maximum number of tokens allowed to generate. smaller = Control cost.
    "max_tokens": None, 
    
    # 0 = Prefer high-probability tokens (more deterministic output). higher = Allow to try lower-probability tokens (more diverse output).
    "temperature": None,
    # Select tokens in descending probability until the cumulative probability ≥ top_p. 1 = Keep all tokens.
    "top_p": None,  
    # Return logits for all tokens, not just the last token.
    "logits_all": False,
    # The number of logprobs to return. 
    "logprobs": None,

    # Verbose logging for debugging and tracing model calls.
    "verbose": False
}

LLAMACPP_CONFIG = {
    "model_path": None,

    # Maximum number of tokens allowed to generate. small = control cost.
    "max_tokens": 256,
    # Context window. The default value is 512.
    "n_ctx": 512,

    # 0 = favor high-probability tokens (more deterministic output). higher = allow to try lower-probability tokens (more diverse output).
    "temperature": 0.8,
    # Select tokens in descending probability until the cumulative probability ≥ top_p. 1 = Keep all tokens.
    "top_p": 0.95,   
    # Return logits for all tokens, not just the last token.
    "logits_all": False,
    # The number of logprobs to return. 
    "logprobs": None,

    # CPU threads.
    "n_threads": None,
    # -1 = offload all layers to GPU.
    "n_gpu_layers": None,

    # Verbose logging for debugging and tracing model calls.
    "verbose": True
}

OLLAMA_CONFIG = {
    "model": None,  

    "num_predict": None,
    "num_ctx": 2048,

    "temperature": 0.8
}