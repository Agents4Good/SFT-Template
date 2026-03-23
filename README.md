# 🚀 SFT Template - Supervised Fine-Tuning with Unsloth

This project provides a professional, modular, and configurable template for Supervised Fine-Tuning (SFT) of Large Language Models (LLMs). It is specifically optimized to leverage **Unsloth**, enabling faster and more memory-efficient training on Consumer or Datacenter GPUs.

---

## ✨ Key Features

- **Modular Design**: Clean separation between Model handlers, Dataset processors, and Training loops.
- **Config-Driven Workflow**: Manage all hyperparameters—from Model quantization to PEFT/LoRA settings—via a single `src/config.yaml` file.
- **Abstract Base Classes**: Built on extensible `ABC` interfaces, allowing for easy integration of new dataset formats or custom training logic.
- **Unsloth Optimization**: Automated integration for 4-bit loading, LoRA/PEFT application, and memory-efficient training.
- **Reproducibility**: Each run is timestamped and automatically backs up its configuration and dataset for future reference.
- **Scaling with Environment Variables**: Override any training parameter (e.g., `LEARNING_RATE`, `NUM_TRAIN_EPOCHS`) on-the-fly via system environment variables.

---

## 📂 Project Structure

```text
.
├── src/
│   ├── core/
│   │   └── base.py           # Abstract Base Classes (Standards Definition)
│   ├── model/
│   │   └── handler.py        # Model/Tokenizer initialization (Unsloth/PEFT)
│   ├── dataset/
│   │   └── processor.py      # Data loading, cleaning, and LoRA formatting
│   ├── train/
│   │   └── sft.py            # Training orchestration and persistence
│   ├── prompts/              # Centralized prompt templates
│   ├── config.yaml           # Centralized configuration file
│   └── main.py              # Application Entry Point
├── data/                    # Recommended directory for raw datasets
├── requirements.txt         # Project dependencies
└── README.md                # Documentation (You are here)
```

---

## 🛠️ Installation

### Prerequisites
- Python 3.10+
- CUDA-compatible GPU

### Setup
We recommend using a Virtual Environment (Conda or venv):

```bash
# Example using Conda
conda create -n sft_template python=3.10 -y
conda activate sft_template

# Install core dependencies
pip install unsloth trl transformers datasets torch pyyaml
```

---

## ⚙️ Configuration & Usage

### 1. Configure the Run
Edit `src/config.yaml` to suit your training needs. Key sections include:

- **`model`**: Choose your base model (e.g., Gemma, Llama 3) and sequence length.
- **`dataset`**: Provide the path to your JSONL file and define train/test splits.
- **`training`**: Standard HF/TRL training arguments.
- **`peft`**: LoRA rank (`r`), alpha, and dropout.

### 2. Execution
Run the training script via the CLI:

```bash
python src/main.py --config src/config.yaml
```

### 3. Dynamic Overrides (Advanced)
You can override any parameter in the `training` section by prefixing the name with its uppercase version in the environment:

```bash
# Override learning rate and epochs without changing the YAML
LEARNING_RATE=5.0e-6 NUM_TRAIN_EPOCHS=10 python src/main.py
```

---

## 🧩 Extending the Template

### Custom Datasets
If your dataset format differs from the default JSONL structure, simply extend `AbstractDatasetProcessor` in `src/dataset/processor.py`:

```python
class MyCustomProcessor(AbstractDatasetProcessor):
    def format_dataset(self, dataset):
        # Your custom mapping logic here
        ...
```

### Custom Prompts
Add new templates to `src/prompts/` to maintain a clean separation between prompting strategies and data logic.

---

## 📈 Monitoring
Logs are saved to the directory specified in `output.output_dir` (default: `./outputs/`). By default, the project supports **Tensorboard** reporting.

---
