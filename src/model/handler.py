from unsloth import FastLanguageModel
from src.core.base import AbstractModel
import torch

class Model(AbstractModel):
    def __init__(self, config):
        self.config = config
        self.model = None
        self.tokenizer = None

    def load_model(self):
        max_seq_length = self.config['model']['max_seq_length']
        dtype = getattr(torch, self.config['model']['dtype']) if hasattr(torch, self.config['model']['dtype']) else None
        load_in_4bit = self.config['model']['load_in_4bit']
        model_name = self.config['model']['name']

        # Load Pre-trained Model and Tokenizer
        self.model, self.tokenizer = FastLanguageModel.from_pretrained(
            model_name=model_name,
            max_seq_length=max_seq_length,
            dtype=dtype,
            load_in_4bit=load_in_4bit,
        )

        # Apply PEFT (LoRA) configuration
        self.model = FastLanguageModel.get_peft_model(
            self.model,
            r=self.config['peft']['r'],
            lora_alpha=self.config['peft']['r'],
            lora_dropout=self.config['peft']['lora_dropout'],
            bias=self.config['peft']['bias'],
            use_gradient_checkpointing=self.config['peft']['use_gradient_checkpointing'],
            random_state=self.config['peft']['random_state'],
        )
    
    def get_model(self):
        """Returns the loaded model instance."""
        return self.model
    
    def get_tokenizer(self):
        """Returns the loaded tokenizer instance."""
        return self.tokenizer
