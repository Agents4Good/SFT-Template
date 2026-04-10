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

def optimize_training_params(vram_metrics, target_vram_gb, global_batch_size):
    """
    Calculates the optimal micro-batch size and gradient accumulation steps.
    
    Args:
        vram_metrics (dict): Metrics from estimate_vram_usage.
        target_vram_gb (float): Target VRAM usage in GB.
        global_batch_size (int): Desired global batch size.
        
    Returns:
        tuple: (per_device_train_batch_size, gradient_accumulation_steps)
    """
    static_vram_gb = vram_metrics["total_static_gb"]
    activation_per_sample_gb = vram_metrics["activations_per_sample_gb"]
    
    available_vram_gb = target_vram_gb - static_vram_gb
    
    if available_vram_gb <= 0:
        # Static part alone exceeds budget or very close to it.
        # Fall back to minimum micro-batch size.
        return 1, global_batch_size
        
    # Calculate how many samples we can fit in the remaining VRAM
    max_per_device_batch = int(available_vram_gb // activation_per_sample_gb)
    
    if max_per_device_batch >= global_batch_size:
        return global_batch_size, 1
        
    # Ensure at least 1
    max_per_device_batch = max(1, max_per_device_batch)
    
    # Find the largest divisor of global_batch_size that is <= max_per_device_batch
    per_device_train_batch_size = 1
    for d in range(max_per_device_batch, 0, -1):
        if global_batch_size % d == 0:
            per_device_train_batch_size = d
            break
    
    # Calculate gradient accumulation steps
    gradient_accumulation_steps = global_batch_size // per_device_train_batch_size
    
    return per_device_train_batch_size, gradient_accumulation_steps
