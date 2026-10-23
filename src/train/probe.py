"""Measured GPU probes on the real training path."""

import shutil
import tempfile
import time

from src.train.fit_check import FitError
from src.train.trainer import close_trainer


def _is_oom(exc):
    return "out of memory" in str(exc).lower()


def usable_memory_gb():
    """Memory this process can still use, and the device total."""
    import torch

    if not torch.cuda.is_available():
        raise FitError("Pre-flight sizing needs a CUDA GPU.")
    torch.cuda.synchronize()
    torch.cuda.empty_cache()
    free, total = torch.cuda.mem_get_info()
    reserved = torch.cuda.memory_reserved()
    usable = (free + reserved) / 1024**3
    if usable <= 0:
        raise FitError("No usable CUDA memory reported.")
    return usable, total / 1024**3


def _probe_eval_batch(trainer):
    """Longest eval example, collated exactly like real eval."""
    dataset = trainer.eval_dataset
    if dataset is None or len(dataset) == 0:
        return None
    try:
        lengths = [len(row["input_ids"]) for row in dataset]
        index = max(range(len(lengths)), key=lengths.__getitem__)
        return trainer.data_collator([dataset[index]])
    except Exception:
        return next(iter(trainer.get_eval_dataloader()))


class ProbeSession:
    """Owns one disposable probe Trainer and its temporary directory."""

    def __init__(self, trainer, tmp_dir, steps):
        self.trainer = trainer
        self.tmp_dir = tmp_dir
        self.steps = steps
        self.probes = 0
        self._released = False
        self._closed = False

    @classmethod
    def open(cls, trainer_factory, steps=3):
        """Create a probe Trainer; `trainer_factory(tmp_dir, steps)`."""
        tmp_dir = tempfile.mkdtemp(prefix="sft-probe-")
        try:
            trainer = trainer_factory(tmp_dir, steps)
        except Exception:
            shutil.rmtree(tmp_dir, ignore_errors=True)
            raise
        return cls(trainer, tmp_dir, steps)

    def _check_usable(self):
        if self._closed:
            raise RuntimeError("ProbeSession is closed.")
        if self._released:
            raise RuntimeError("ProbeSession trainer was released.")

    def _clear_grads(self):
        model = getattr(self.trainer, "model", None)
        if model is not None:
            try:
                for param in model.parameters():
                    if param.grad is not None:
                        param.grad = None
            except Exception:
                pass
        try:
            import torch

            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        except Exception:
            pass

    def run(self, micro):
        """Run one measured probe at `micro` on the reusable session."""
        self._check_usable()
        import torch

        trainer = self.trainer
        trainer.args.per_device_train_batch_size = micro
        trainer._train_batch_size = micro
        trainer.args.gradient_accumulation_steps = 1
        trainer.control.should_training_stop = False
        torch.cuda.synchronize()
        # Each probe starts clean: drop cached blocks from earlier probes
        # so the peak reflects this candidate, not stale reservations.
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()
        started = time.monotonic()
        try:
            trainer.train()
            batch = _probe_eval_batch(trainer)
            if batch is not None:
                trainer.model.eval()
                with torch.no_grad(), trainer.compute_loss_context_manager():
                    trainer.model(**trainer._prepare_inputs(batch))
            torch.cuda.synchronize()
            peak = torch.cuda.max_memory_reserved() / 1024**3
        except RuntimeError as exc:
            if not _is_oom(exc):
                raise
            raise FitError(
                f"Trainer probe OOM at micro={micro} "
                f"(probe_steps={self.steps})."
            ) from exc
        finally:
            self._clear_grads()
            trainer.model.train()
            torch.cuda.empty_cache()
        self.probes += 1
        return {
            "peak_gb": peak,
            "duration_s": round(time.monotonic() - started, 2),
            "steps": self.steps,
        }

    def release_trainer(self):
        """Hand the Trainer to the caller; this session must not touch it again."""
        self._check_usable()
        self._released = True
        return self.trainer

    def close(self):
        """Tear down the session; safe to call more than once."""
        if self._closed:
            return
        self._closed = True
        if not self._released:
            close_trainer(self.trainer)
        self.trainer = None
        shutil.rmtree(self.tmp_dir, ignore_errors=True)

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc, traceback):
        self.close()
