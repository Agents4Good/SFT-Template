from trl import SFTTrainer
from transformers import TrainingArguments, EarlyStoppingCallback
from core.base import AbstractSFTTrain
import datetime
import os
import ast
import shutil

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
                except:
                    training_config[variable] = environment_value

                print(f"Overriding {variable} with environment variable: {training_config[variable]}")

        # Finalize path again to ensure accuracy at training start
        self.timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        self.final_path = f"{self.config['output']['output_dir']}/run-{self.timestamp}"

        # Setup directory structure and metadata backups
        os.makedirs(self.final_path, exist_ok=True)
        if self.config['output']['save_config']:
            shutil.copy2('./src/config.yaml', os.path.join(self.final_path, 'config.yaml'))
        
        # Only try to copy the dataset if it refers to a local path
        if self.config['output']['save_dataset'] and os.path.exists(self.config['dataset']['path']):
            shutil.copy2(self.config['dataset']['path'], self.final_path)
        
        # Initialize training arguments from configuration
        args = TrainingArguments(
            output_dir=self.final_path,
            overwrite_output_dir=True,
            per_device_train_batch_size=training_config['per_device_train_batch_size'],
            gradient_accumulation_steps=training_config['gradient_accumulation_steps'],
            optim=training_config['optim'],
            learning_rate=float(training_config['learning_rate']),
            weight_decay=training_config['weight_decay'],
            lr_scheduler_type=training_config['lr_scheduler_type'],
            warmup_steps=training_config['warmup_steps'],
            logging_steps=training_config['logging_steps'],
            num_train_epochs=training_config['num_train_epochs'],
            max_steps=training_config['max_steps'],
            fp16=training_config['fp16'],
            bf16=training_config['bf16'],
            eval_strategy=training_config['eval_strategy'],
            eval_steps=training_config['eval_steps'],
            save_strategy=training_config['save_strategy'],
            save_steps=training_config['save_steps'],
            load_best_model_at_end=training_config['load_best_model_at_end'],
            metric_for_best_model=training_config['metric_for_best_model'],
            push_to_hub=training_config['push_to_hub'],
            report_to=training_config['report_to'],
            gradient_checkpointing=self.config['peft']['use_gradient_checkpointing'],
            max_grad_norm=training_config['max_grad_norm'],
        )

        callbacks = []
        if training_config.get('early_stopping', {}).get('enabled', False):
            callbacks.append(EarlyStoppingCallback(
                early_stopping_patience=training_config['early_stopping']['patience'],
                early_stopping_threshold=training_config['early_stopping']['threshold']
            ))

        # Build SFTTrainer
        self.trainer = SFTTrainer(
            model=self.model,
            tokenizer=self.tokenizer,
            train_dataset=self.train_dataset,
            eval_dataset=self.eval_dataset,
            dataset_text_field="text",
            max_seq_length=self.config['model']['max_seq_length'],
            args=args,
            packing=False,
            callbacks=callbacks,
        )

        # Execute training
        self.trainer.train()

    def save_model(self): 
        """
        Saves the trainer's model weights to the final destination directory.
        """
        self.trainer.save_model(self.final_path + "/Model")
        print(f"Model saved successfully to {self.final_path}!")
