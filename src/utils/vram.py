import torch

def estimate_vram_usage(
    model_name: str,
    total_params: int,
    trainable_params: int,
    dtype: str,
    load_in_4bit: bool,
    max_seq_length: int,
    optimizer: str = "adamw_8bit"
):
    """
    Estimates VRAM usage per unit (sample) during training.
    
    Args:
        model_name (str): Name of the model.
        total_params (int): Total number of model parameters.
        trainable_params (int): Number of trainable parameters (LoRA).
        dtype (str): Data type for training (float16, bfloat16, float32).
        load_in_4bit (bool): Whether the model is loaded in 4-bit.
        max_seq_length (int): Maximum sequence length.
        optimizer (str): Optimizer name (e.g., 'adamw_8bit').
        
    Returns:
        dict: Estimated memory components in GB.
    """
    # Dtype handling: Address specific fallbacks if necessary.
    # Note: Unsloth sometimes forces float32 for certain models like Gemma 3 if float16 is requested.
    effective_dtype = dtype
    if "gemma-3" in model_name.lower() and dtype == "float16":
        effective_dtype = "float32"
    
    bytes_per_param = {
        "float32": 4,
        "float16": 2,
        "bfloat16": 2,
    }.get(effective_dtype, 4)

    # 1. Weights Memory
    if load_in_4bit:
        # 4-bit quantization approx 0.5 bytes + overhead (~0.2)
        weight_vram = total_params * 0.7 
    else:
        weight_vram = total_params * bytes_per_param
    
    # 2. Gradients Memory (Trainable parameters)
    grad_vram = trainable_params * bytes_per_param
    
    # 3. Optimizer Memory (Trainable parameters)
    if "8bit" in optimizer:
        opt_vram = trainable_params * 2
    else:
        # Standard Adam/AdamW uses 8 bytes per trainable parameter for moments
        opt_vram = trainable_params * 8
        
    # 4. Activation Memory (Rough estimate per unit sample)
    # Heuristic: ~2 * layers * hidden_size * seq_len * bytes
    # For many models, total_params is approx 12 * layers * hidden^2.
    # We can approximate hidden_size * layers as a factor of total_params.
    # A safe rule of thumb for gradient checkpointing is ~0.1-0.5 MB per token for 1B-7B models.
    activation_vram_per_sample = max_seq_length * (total_params / 1e6) * 0.4 * 1024
    
    # Convert to GB
    to_gb = lambda x: round(x / (1024**3), 3)
    
    metrics = {
        "weights_gb": to_gb(weight_vram),
        "gradients_gb": to_gb(grad_vram),
        "optimizer_gb": to_gb(opt_vram),
        "activations_per_sample_gb": to_gb(activation_vram_per_sample),
        "total_static_gb": to_gb(weight_vram + grad_vram + opt_vram),
        "total_per_sample_gb": to_gb(weight_vram + grad_vram + opt_vram + activation_vram_per_sample)
    }
    
    return metrics
