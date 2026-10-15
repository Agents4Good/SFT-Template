from trl import SFTTrainer, SFTConfig
from transformers import EarlyStoppingCallback
from src.core.base import AbstractSFTTrain
from src.utils.push import push_model
import datetime
import os
import ast
import shutil


class LengthGroupedSFTTrainer(SFTTrainer):
    """SFTTrainer that batches similar-length examples together.

    transformers 5.x removed the group_by_length flag but still ships
    LengthGroupedSampler; this wires it through the _get_train_sampler
    extension point. Example boundaries and loss are unchanged, only
    which examples share a batch differs, so padding waste shrinks.
    """

    def __init__(self, *args, group_by_length=False, **kwargs):
        self._group_by_length = group_by_length
        super().__init__(*args, **kwargs)

    def _get_train_sampler(self, train_dataset=None):
        if not self._group_by_length:
            return super()._get_train_sampler(train_dataset)
        from transformers.trainer_pt_utils import LengthGroupedSampler

        return LengthGroupedSampler(
            self.args.train_batch_size
            * self.args.gradient_accumulation_steps,
            dataset=train_dataset
            if train_dataset is not None else self.train_dataset,
            model_input_name="input_ids",
        )

class SFTTrain(AbstractSFTTrain):
    """
    Handles the Supervised Fine-Tuning (SFT) training cycle.
    
    This class manages:
    - Overriding training parameters via environment variables.
    - Setting up output directories with timestamp suffixes.
    - Configuring the Trainer object from the Transformers/TRL library.
    - Executing training and saving the final model.
    """

    def __init__(self, config, model, tokenizer, train_dataset, eval_dataset):
        """
        Initializes the SFT trainer wrapper.
        
        Args:
            config (dict): The project configuration.
            model: The PEFT-wrapped model instance.
            tokenizer: The model's tokenizer.
            train_dataset: The processed training dataset.
            eval_dataset: The processed evaluation dataset.
        """
        self.config = config
        self.model = model
        self.tokenizer = tokenizer
        self.train_dataset = train_dataset
        self.eval_dataset = eval_dataset
        self.trainer = None
        # Unique timestamp for each training run
        self.timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        self.final_path = f"{self.config['output']['output_dir']}/run-{self.timestamp}"

    def train(self):
        """
        Prepares and executes the training process.
        - Applies environment variable overrides if present (e.g., LEARNING_RATE).
        - Creates the output directory and backups config/data.
        - Initializes the SFTTrainer and starts the loop.
        """
        training_config = self.config['training']
        
        # Override config parameters with environment variables for flexible scaling
        for variable in training_config.keys():
            environment_value = os.getenv(variable.upper())

            if environment_value:
                try:
                    # Parse numerical or complex types using literal_eval
                    training_config[variable] = ast.literal_eval(environment_value)
                except (ValueError, SyntaxError):
                    training_config[variable] = environment_value

                print(f"Overriding {variable} with environment variable: {training_config[variable]}")

        # Batch sizes arrive resolved (search or pinned in main.py, env
        # overrides above): they must be present and positive here.
        micro = training_config.get('per_device_train_batch_size', 0)
        accum = training_config.get('gradient_accumulation_steps', 0)
        if (not isinstance(micro, int) or not isinstance(accum, int)
                or micro <= 0 or accum <= 0):
            raise ValueError(
                "training.per_device_train_batch_size and "
                "training.gradient_accumulation_steps must be positive "
                f"ints, got {micro!r} and {accum!r}.")

        # Finalize path again to ensure accuracy at training start
        self.timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        self.final_path = f"{self.config['output']['output_dir']}/run-{self.timestamp}"

        # Setup directory structure and metadata backups
        os.makedirs(self.final_path, exist_ok=True)
        if self.config['output']['save_config']:
            import yaml as _yaml
            # Persist the live config (includes env overrides) rather
            # than the stale on-disk file.
            with open(os.path.join(self.final_path, 'config.yaml'), 'w') as f:
                _yaml.safe_dump(
                    {k: v for k, v in self.config.items() if not k.startswith('_')},
                    f,
                )
        
        # Only try to copy the dataset if it refers to a local path
        if self.config['output']['save_dataset'] and os.path.exists(self.config['dataset']['path']):
            shutil.copy2(self.config['dataset']['path'], self.final_path)
        
        # Deduce precision from dtype
        dtype_str = self.config['model'].get('dtype', 'float32')
        fp16_val = dtype_str == 'float16'
        bf16_val = dtype_str == 'bfloat16'
        
        # Training arguments (pinned stack: trl 1.13, transformers 5.17).
        args = SFTConfig(
            output_dir=self.final_path,
            dataset_text_field="text",
            max_length=self.config['model']['max_seq_length'],
            per_device_train_batch_size=training_config['per_device_train_batch_size'],
            # Eval batch pinned to 1, decoupled from the train micro: real
            # runs showed every allocator OOM-retry comes from early-eval
            # transient grabs at larger eval micros; train itself was
            # retry-free at every micro 1-4. The HF default (8) OOMs
            # large models at eval time.
            per_device_eval_batch_size=1,
            gradient_accumulation_steps=training_config['gradient_accumulation_steps'],
            save_total_limit=training_config.get('save_total_limit', 2),
            optim=training_config['optim'],
            learning_rate=float(training_config['learning_rate']),
            weight_decay=0.01,
            lr_scheduler_type=training_config['lr_scheduler_type'],
            warmup_steps=training_config['warmup_steps'],
            logging_steps=10,
            num_train_epochs=training_config['num_train_epochs'],
            max_steps=training_config['max_steps'],
            fp16=fp16_val,
            bf16=bf16_val,
            eval_strategy=training_config['eval_save_strategy'],
            eval_steps=training_config['eval_save_steps'],
            save_strategy=training_config['eval_save_strategy'],
            save_steps=training_config['eval_save_steps'],
            load_best_model_at_end=training_config['load_best_model_at_end'],
            metric_for_best_model=training_config['metric_for_best_model'],
            push_to_hub=training_config['push_to_hub'],
            report_to=training_config['report_to'],
            gradient_checkpointing=self.config['peft']['use_gradient_checkpointing'],
        )

        callbacks = []
        if training_config.get('early_stopping', {}).get('enabled', False):
            callbacks.append(EarlyStoppingCallback(
                early_stopping_patience=training_config['early_stopping']['patience'],
                early_stopping_threshold=training_config['early_stopping']['threshold']
            ))

        self.trainer = LengthGroupedSFTTrainer(
            model=self.model,
            train_dataset=self.train_dataset,
            eval_dataset=self.eval_dataset,
            args=args,
            callbacks=callbacks,
            processing_class=self.tokenizer,
            group_by_length=training_config.get('group_by_length', False),
        )

        # Execute training
        self.trainer.train()

        # Post-run actuals: peak reserved/allocated + arch fingerprint go
        # to peak_memory.json. Observability only -- nothing here feeds
        # any decision (there is no sizing anymore). Best-effort: never
        # fail the run on telemetry.
        try:
            import torch as _torch

            actuals = {
                "backend": getattr(self.model, "__class__", type).__name__,
                "per_device_train_batch_size": training_config[
                    "per_device_train_batch_size"],
                "gradient_accumulation_steps": training_config[
                    "gradient_accumulation_steps"],
            }
            if _torch.cuda.is_available():
                actuals["actual_peak_reserved_gb"] = round(
                    _torch.cuda.max_memory_reserved() / 1024**3, 3
                )
                actuals["actual_peak_allocated_gb"] = round(
                    _torch.cuda.max_memory_allocated() / 1024**3, 3
                )
            fingerprint = {}
            model_cfg_obj = getattr(self.model, "config", None)
            text_cfg = (
                getattr(model_cfg_obj, "text_config", None)
                or model_cfg_obj
            )
            for name in (
                "model_type", "hidden_size", "num_hidden_layers",
                "num_attention_heads", "num_key_value_heads",
                "vocab_size", "sliding_window",
            ):
                for obj in (model_cfg_obj, text_cfg):
                    if obj is not None and hasattr(obj, name):
                        try:
                            fingerprint[name] = getattr(obj, name)
                            break
                        except Exception:
                            continue
            fingerprint["attn_implementation"] = (
                getattr(model_cfg_obj, "attn_implementation", None)
                if model_cfg_obj is not None else None
            )
            fingerprint["gradient_checkpointing"] = self.config.get(
                "peft", {}).get("use_gradient_checkpointing")
            try:
                import cut_cross_entropy  # noqa: F401

                fingerprint["chunked_loss"] = True
            except Exception:
                fingerprint["chunked_loss"] = False
            actuals["arch_fingerprint"] = fingerprint
            search = self.config.get("_micro_search")
            if search:
                actuals["micro_search"] = search
            import json as _json

            with open(os.path.join(self.final_path,
                                   "peak_memory.json"), "w") as f:
                _json.dump(actuals, f, indent=2, default=str)
        except Exception as e:
            print(f"WARNING: post-run VRAM telemetry failed ({e}); "
                  "continuing without actuals.")

    def save_model(self): 
        """
        Saves the trainer's model weights to the final destination directory.
        """
        model_dir = self.final_path + "/Model"
        self.trainer.save_model(model_dir)
        print(f"Model saved successfully to {self.final_path}!")
        push_cfg = (self.config.get("output") or {}).get("push") or {}
        if push_cfg.get("enabled", False):
            push_model(
                model_dir,
                push_cfg.get("repo_id"),
                revision=push_cfg.get("revision", "main"),
                private=push_cfg.get("private", False),
            )
