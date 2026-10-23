"""SFT orchestration: run artifacts, the training loop, and save/push."""

import ast
import datetime
import os
import shutil

from transformers import set_seed

from src.core.base import AbstractSFTTrain
from src.train import telemetry
from src.utils.push import push_model


class SFTTrain(AbstractSFTTrain):
    """Runs a trainer that has already been sized and promoted.

    This class manages:
    - Overriding training parameters via environment variables.
    - Setting up output directories with timestamp suffixes.
    - Executing training and saving the final model.
    """

    def __init__(self, config, model, tokenizer, train_dataset, eval_dataset):
        self.config = config
        self.model = model
        self.tokenizer = tokenizer
        self.train_dataset = train_dataset
        self.eval_dataset = eval_dataset
        self.trainer = None
        self._env_applied = False
        self.timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        self.final_path = f"{config['output']['output_dir']}/run-{self.timestamp}"

    def apply_env_overrides(self):
        """Apply environment-variable overrides once, in place."""
        training_config = self.config['training']
        if self._env_applied:
            return training_config

        for variable in training_config.keys():
            environment_value = os.getenv(variable.upper())

            if environment_value:
                try:
                    training_config[variable] = ast.literal_eval(
                        environment_value)
                except (ValueError, SyntaxError):
                    training_config[variable] = environment_value

                print(f"Overriding {variable} with environment variable: "
                      f"{training_config[variable]}")
        self._env_applied = True
        return training_config

    @staticmethod
    def _validate_batch(training):
        micro = training.get('per_device_train_batch_size', 0)
        accum = training.get('gradient_accumulation_steps', 0)
        if (not isinstance(micro, int) or not isinstance(accum, int)
                or micro <= 0 or accum <= 0):
            raise ValueError(
                "training.per_device_train_batch_size and "
                "training.gradient_accumulation_steps must be positive "
                f"ints, got {micro!r} and {accum!r}.")

    def _prepare_run_dir(self):
        os.makedirs(self.final_path, exist_ok=True)
        if self.config['output']['save_config']:
            import yaml

            # Persist the live config (includes env overrides) rather
            # than the stale on-disk file.
            with open(os.path.join(self.final_path, 'config.yaml'), 'w') as f:
                yaml.safe_dump(
                    {k: v for k, v in self.config.items()
                     if not k.startswith('_')},
                    f,
                )

        # Only try to copy the dataset if it refers to a local path
        if (self.config['output']['save_dataset']
                and os.path.exists(self.config['dataset']['path'])):
            shutil.copy2(self.config['dataset']['path'], self.final_path)

    def train(self, trainer, preflight=None):
        """Train `trainer` and record the run's measured actuals."""
        training_config = self.apply_env_overrides()
        self._validate_batch(training_config)
        self._prepare_run_dir()

        self.trainer = trainer
        import torch

        if torch.cuda.is_available():
            torch.cuda.reset_peak_memory_stats()
        set_seed(self.trainer.args.seed)
        self.trainer.train()

        telemetry.record_training_actuals(
            self.final_path, self.model, self.config, preflight=preflight)

    def save_model(self):
        """Save the trained weights and optionally push them to the Hub."""
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
