# Multimodal Sarcasm Detection with Certainty Estimation

![Python](https://img.shields.io/badge/Python-3.8%2B-blue)
![PyTorch](https://img.shields.io/badge/PyTorch-2.0%2B-ee4c2c)
![License](https://img.shields.io/badge/License-MIT-green)

## 📖 Introduction

This project explores multimodal sarcasm detection in a Chinese context. Unlike traditional binary classification tasks, this project introduces a **Multi-Task Learning (MTL)** architecture that simultaneously predicts sarcasm labels and regresses sarcasm certainty/intensity levels (1-not sure, 2-almost sure, 3-very sure).

The baseline model employs an offline feature extraction with a late fusion strategy, ensuring experimental stability and high reproducibility.


## 📁 Porject Structure

mlt_sarcasm/
├── data/
│   ├── preprocess.py          # Data preprocessing & feature extraction scripts
│   └── dataset.py             # Custom PyTorch dataset classes for multimodal data
├── models/
│   ├── base_model.py          # Base model architecture & modality encoders
│   └── mtl_model.py           # Multi-task learning model (Late Fusion & Dual Heads)
├── training/
│   ├── trainer.py             # Training loop, loss computation, and optimization
│   └── evaluation.py          # Model evaluation metrics (F1, Accuracy, MAE, Pearson)
├── utils/
│   ├── config.py              # Configuration parsing and management
│   └── logger.py              # Logging utilities (TensorBoard/W&B integration)
├── configs/                   
│   └── default.yaml           # Default hyperparameters and path configurations
├── main.py                    # Main execution script
└── requirements.txt           # Python dependencies


## 🛠 Environment Setup

We recommend using Conda to manage your Python environment:

```bash
conda create -n sarcasm_mtl python==3.10
conda activate sarcasm_mtl
pip install -r requirements.txt
```


## 🚀 Usage

### Training (Text + Audio + Video)
To train the model, run the `main.py` script:
```bash
python -m src.trainer.train \
  --exp_name Text_Audio_Video \
  --model_type collabrative \
  --lambda_reg 0.1
```

### Training (Text Only)
To train the model, run the `main.py` script:
```bash
python -m src.trainer.train \
  --exp_name Text_Only \
  --model_type gating \
  --lambda_reg 0.1 \
  --ablate_audio \
  --ablate_video
```

The training script will automatically load the configuration from `configs/default.yaml`. You can modify the configuration by editing this file or by passing a custom configuration file using the `--config` argument.



## Results
The results of the model can be found in the `output` directory.