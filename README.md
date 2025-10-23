# MegaDescriptor Animal Re-ID Fine-tuning

A streamlined framework for fine-tuning MegaDescriptor-T-224 model on custom animal re-identification datasets using `wildlife-tools` and `wildlife-datasets`.

## Quick Start

### 1. Setup Project
```

# Install dependencies
pip install -r requirements.txt
pip install wildlife-datasets
pip install git+https://github.com/WildlifeDatasets/wildlife-tools
```

### 2. Prepare Data
Organize your data:
```
data/raw/my_animal_dataset/
    ├── individual_001/
    │   ├── image_001.nef
    │   └── ...
    ├── individual_002/
    │   ├── image_001.jpg
    │   └── ...
```

Validate data:
```bash
python scripts/prepare_data.py --data_root data/raw/my_animal_dataset
```

### 3. Train Model
```bash
python scripts/train.py --config configs/train_megadescriptor.yaml
```

### 4. Evaluate Model
```bash
python scripts/evaluate.py --config configs/train_megadescriptor.yaml --checkpoint outputs/checkpoints/final_model.pth
```

## Features

- **Custom Dataset Support**: Handles folder-based animal datasets
- **Pre-trained MegaDescriptor**: Uses MegaDescriptor-T-224 from HuggingFace
- **ArcFace Loss**: Optimized for re-identification tasks
- **Data Augmentation**: Comprehensive training transformations
- **Training Monitoring**: Automatic checkpointing and logging

## Configuration

Edit `configs/train_megadescriptor.yaml` to adjust:
- Training parameters (epochs, batch size, learning rate)
- Data augmentation settings
- Model and output paths

## Project Structure
```
mega_descriptor_finetune/
├── configs/              # Configuration files
├── data/                 # Data directories
├── src/                  # Source code
├── scripts/              # Execution scripts
├── outputs/              # Training outputs
├── requirements.txt      # Dependencies
└── README.md            # This file
```

## Requirements

- Python 3.8+
- PyTorch 1.9+
- wildlife-datasets, wildlife-tools
- Optional: rawpy for .NEF file support

## Outputs

- **Checkpoints**: `outputs/checkpoints/`
- **Logs**: `outputs/logs/`
- **Visualizations**: `outputs/visualizations/`

For detailed documentation, see the code comments and configuration files.