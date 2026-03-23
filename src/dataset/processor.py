from datasets import load_dataset, Dataset
from core.base import AbstractDatasetProcessor
from prompts import user_prompt, model_prompt

class DatasetProcessor(AbstractDatasetProcessor):
    def __init__(self, config, tokenizer):
        self.config = config
        self.tokenizer = tokenizer

    def load_dataset(self, path=None):
        dataset_path = path if path else self.config['dataset']['path']
        dataset = load_dataset(dataset_path, "main", split="train")
        df = dataset.to_pandas()
        df['id'] = range(len(df))
        df['trace'] = df['answer']
        df['raw_answer'] = df['answer']
        return Dataset.from_pandas(df)

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
        
        cleaned_dataset = dataset.filter(lambda example: example['answer'] != '*')
        cleaned_dataset = cleaned_dataset.filter(lambda example: example['id'] not in {144, 190})
        
        return cleaned_dataset.map(formatting_prompts_func)

    def split_dataset(self, dataset):
        split_config = self.config['dataset']['train_test_split']
        return dataset.train_test_split(
            test_size=split_config['test_size'], 
            shuffle=split_config['shuffle'], 
            seed=split_config['seed']
        )

    def save_dataset(self, dataset, path):
        dataset.save_to_disk(path)
