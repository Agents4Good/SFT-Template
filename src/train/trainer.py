"""Trainer construction, probe-to-real promotion and disposal."""

import gc

from trl import SFTConfig, SFTTrainer
from transformers import EarlyStoppingCallback
from transformers.trainer_pt_utils import LengthGroupedSampler


class LengthGroupedSFTTrainer(SFTTrainer):
    """SFTTrainer with explicit length-grouped sampling.

    transformers infers lengths from processing_class.model_input_names[0],
    which on multimodal processors is "pixel_values", not "input_ids". The
    trainer here is handed a processor, so inference breaks; feeding the
    sampler measured token counts keeps grouped batching (longest first,
    the real worst case for VRAM) working.
    """

    def _explicit_lengths(self, dataset):
        column = getattr(self.args, "length_column_name", "length")
        if column in dataset.column_names:
            return list(dataset[column])
        return [len(ids) for ids in dataset["input_ids"]]

    def _get_train_sampler(self, train_dataset=None):
        if train_dataset is None:
            train_dataset = self.train_dataset
        strategy = getattr(self.args, "train_sampling_strategy", "random")
        if train_dataset is None or strategy != "group_by_length":
            return super()._get_train_sampler(train_dataset)
        return LengthGroupedSampler(
            self.args.train_batch_size
            * self.args.gradient_accumulation_steps,
            lengths=self._explicit_lengths(train_dataset),
        )

    def _get_eval_sampler(self, eval_dataset):
        strategy = getattr(self.args, "train_sampling_strategy", "random")
        if eval_dataset is None or strategy != "group_by_length":
            return super()._get_eval_sampler(eval_dataset)
        import torch

        return LengthGroupedSampler(
            self.args.eval_batch_size,
            lengths=self._explicit_lengths(eval_dataset),
            generator=torch.Generator().manual_seed(
                self.args.data_seed
                if self.args.data_seed is not None else self.args.seed),
        )


def _precision_flags(config):
    dtype_str = config["model"].get("dtype", "float32")
    return dtype_str == "float16", dtype_str == "bfloat16"


def _sampling_strategy(training):
    return ("group_by_length"
            if training.get("group_by_length", False) else "random")


def build_args(config, micro, accum, output_dir, probe=False, probe_steps=3):
    """Build SFTConfig for a real run or for a disposable probe.

    Probe args keep the real stack (optimizer, precision, gradient
    checkpointing, sampler) but disable eval/save/reporting, force lr=0
    so weights stay untouched, and cap the run at a few steps.
    """
    training = config["training"]
    fp16_val, bf16_val = _precision_flags(config)
    kwargs = dict(
        output_dir=output_dir,
        dataset_text_field="text",
        max_length=config["model"]["max_seq_length"],
        per_device_train_batch_size=micro,
        # Eval batch pinned to 1, decoupled from the train micro.
        per_device_eval_batch_size=1,
        gradient_accumulation_steps=accum,
        optim=training["optim"],
        learning_rate=float(training["learning_rate"]),
        weight_decay=0.01,
        lr_scheduler_type=training["lr_scheduler_type"],
        warmup_steps=training["warmup_steps"],
        logging_steps=10,
        num_train_epochs=training["num_train_epochs"],
        max_steps=training["max_steps"],
        fp16=fp16_val,
        bf16=bf16_val,
        push_to_hub=training["push_to_hub"],
        report_to=training["report_to"],
        gradient_checkpointing=(
            config["peft"]["use_gradient_checkpointing"]),
        train_sampling_strategy=_sampling_strategy(training),
    )
    if probe:
        kwargs.update(
            gradient_accumulation_steps=1,
            learning_rate=0.0,
            weight_decay=0.0,
            num_train_epochs=1.0,
            max_steps=probe_steps,
            warmup_steps=0,
            logging_strategy="no",
            eval_strategy="no",
            eval_steps=None,
            save_strategy="no",
            save_steps=None,
            load_best_model_at_end=False,
            metric_for_best_model=None,
            push_to_hub=False,
            report_to="none",
            save_total_limit=None,
            disable_tqdm=True,
            dataloader_num_workers=0,
        )
    else:
        kwargs.update(
            save_total_limit=training.get("save_total_limit", 2),
            eval_strategy=training["eval_save_strategy"],
            eval_steps=training["eval_save_steps"],
            save_strategy=training["eval_save_strategy"],
            save_steps=training["eval_save_steps"],
            load_best_model_at_end=training["load_best_model_at_end"],
            metric_for_best_model=training["metric_for_best_model"],
        )
    return SFTConfig(**kwargs)


def build_callbacks(config):
    """Callbacks owned by the caller (reporting ones come from Trainer)."""
    early = config["training"].get("early_stopping", {})
    if not early.get("enabled", False):
        return []
    return [EarlyStoppingCallback(
        early_stopping_patience=early["patience"],
        early_stopping_threshold=early["threshold"],
    )]


def build_trainer(model, tokenizer, train_dataset, eval_dataset, args,
                  callbacks=()):
    """Build the SFT trainer for one concrete batch configuration."""
    return LengthGroupedSFTTrainer(
        model=model,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        args=args,
        callbacks=list(callbacks),
        processing_class=tokenizer,
    )


def _callback_key(callback):
    return callback if isinstance(callback, type) else type(callback)


def _add_callbacks(trainer, callbacks):
    """Add callbacks once. Integrations arrive as classes, not instances."""
    existing = {_callback_key(callback)
                for callback in trainer.callback_handler.callbacks}
    for callback in callbacks:
        key = _callback_key(callback)
        if key not in existing:
            trainer.add_callback(callback)
            existing.add(key)


def promote_trainer(trainer, real_args, callbacks=()):
    """Turn a probe Trainer into the real training Trainer.

    The probes and the final run share one trainer instance, so compiled
    state and allocations are exactly what was measured. Arguments are
    copied onto the existing `trainer.args` object because the Trainer and
    Accelerate hold references to it.
    """
    args = trainer.args
    for name, value in vars(real_args).items():
        setattr(args, name, value)

    # Trainer caches the batch size at init; Accelerate divides the loss
    # by its own copy at backward time.
    trainer._train_batch_size = args.train_batch_size
    trainer.accelerator.gradient_accumulation_steps = (
        args.gradient_accumulation_steps)

    # Probe moments were accumulated with lr=0: drop them so real
    # training starts from a clean optimizer/scheduler.
    trainer.optimizer = None
    trainer.lr_scheduler = None
    trainer._created_lr_scheduler = False
    trainer.control.should_training_stop = False
    trainer.control.should_log = False
    trainer.control.should_evaluate = False
    trainer.control.should_save = False

    # The probe was built with report_to="none", so integration callbacks
    # were never registered.
    from transformers.integrations import (
        get_reporting_integration_callbacks)

    _add_callbacks(trainer, get_reporting_integration_callbacks(args.report_to))
    _add_callbacks(trainer, callbacks)
    return trainer


def close_trainer(trainer):
    """Drop a Trainer and its accelerate hooks, then free CUDA memory."""
    if trainer is None:
        return
    model = getattr(trainer, "model", None)
    if model is not None:
        try:
            from accelerate.hooks import remove_hook_from_module

            remove_hook_from_module(model, recurse=True)
        except Exception:
            pass
        try:
            for param in model.parameters():
                if param.grad is not None:
                    param.grad = None
        except Exception:
            pass
    gc.collect()
    try:
        import torch

        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    except Exception:
        pass
