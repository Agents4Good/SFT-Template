from datasets import load_dataset, Dataset
from src.core.base import AbstractDatasetProcessor
from src.prompts.train_prompts import user_prompt, model_prompt

DATASET_SUBSET = "main"
DATASET_SPLIT = "train"
EXCLUDE_ANSWERS = {"*"}
EXCLUDE_IDS = {144, 190}


class DatasetProcessor(AbstractDatasetProcessor):
    def __init__(self, config, tokenizer=None):
        self.config = config
        self._tokenizer = tokenizer

    @property
    def tokenizer(self):
        if self._tokenizer is None:
            from transformers import AutoTokenizer

            # Mirror the loader's first attempt: "auto" tries untrusted
            # first; the loader itself retries loudly when custom code
            # is actually needed.
            trc = self.config['model'].get('trust_remote_code', 'auto')
            if not isinstance(trc, bool):
                trc = str(trc).lower() == 'true'
            self._tokenizer = AutoTokenizer.from_pretrained(
                self.config['model']['name'], trust_remote_code=trc)
        return self._tokenizer

    @tokenizer.setter
    def tokenizer(self, value):
        self._tokenizer = value

    def load_dataset(self, path=None):
        dataset_path = path if path else self.config['dataset']['path']
        dataset = load_dataset(dataset_path, DATASET_SUBSET, split=DATASET_SPLIT)
        df = dataset.to_pandas()
        df['id'] = range(len(df))
        df['trace'] = df['answer']
        df['raw_answer'] = df['answer']
        return Dataset.from_pandas(df, preserve_index=False)

    def format_dataset(self, dataset):
        def formatting_prompts_func(example):
            question = example.get('question', '')
            trace = example.get('trace', '')

            prompt_text = user_prompt(question)
            model_response_str = model_prompt(trace)

            messages = [
                {"role": "user", "content": prompt_text},
                {"role": "model", "content": model_response_str},
            ]

            return {"text": self.tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=False)}
        
        cleaned_dataset = dataset.filter(
            lambda example: example['answer'] not in EXCLUDE_ANSWERS)
        cleaned_dataset = cleaned_dataset.filter(
            lambda example: example['id'] not in EXCLUDE_IDS)
        
        return cleaned_dataset.map(formatting_prompts_func)

    def split_dataset(self, dataset):
        split_config = self.config['dataset']['train_test_split']
        return dataset.train_test_split(
            test_size=split_config['test_size'], 
            shuffle=True, 
            seed=42
        )

    def calculate_tokens_usage(self, dataset) -> list:
        # Batched tokenization is much faster than per-example tokenize().
        texts = [ex["text"] for ex in dataset]
        enc = self.tokenizer(
            texts, add_special_tokens=True, truncation=False, padding=False
        )
        return [len(ids) for ids in enc["input_ids"]]

    def suggest_seq_length(
        self, lengths, percentile: float = 95.0, margin: float = 1.1, cap: int = 4096
    ) -> dict:
        """Summarize token lengths and suggest a robust max_seq_length.

        Uses a percentile (not max) so one outlier doesn't blow up VRAM.
        """
        import numpy as np

        arr = np.asarray(lengths, dtype=float)
        stats = {
            "count": int(arr.size),
            "max": int(arr.max()) if arr.size else 0,
            "mean": round(float(arr.mean()), 1) if arr.size else 0,
            f"p{percentile:g}": (
                int(np.percentile(arr, percentile)) if arr.size else 0
            ),
        }
        suggested = (
            min(int(stats[f"p{percentile:g}"] * margin), cap) if arr.size else cap
        )
        stats["suggested"] = suggested
        return stats

