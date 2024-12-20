# TDNN-Conformer designed for the WeNet Framework

Below is a Time Delay Neural Network (TDNN) Conformer implementation for the WeNet speech recognition toolkit.
The TDNN-Conformer model is trained on the LibriSpeech dataset (instructions regarding download are below).

## Table of Contents

- [Prerequisites](#prerequisites)
- [Installation](#installation)
- [Setup](#setup)
- [Training](#training)
- [Evaluation](#evaluation)
- [Configuration](#configuration)
- [Documentation](#documentation)

## Prerequisites

* Python 3.7+
* CUDA-compatible GPU (recommended)
* 50GB+ free disk space
* Git

## Installation

1. Set up virtual environment:
   ```bash
   python -m venv wenet_env
   source wenet_env/bin/activate  # Linux/Mac
   # or
   .\wenet_env\Scripts\activate   # Windows
   ```

2. Clone WeNet:
   ```bash
   git clone https://github.com/wenet-e2e/wenet.git
   cd wenet
   ```

3. Install dependencies:
   ```bash
   pip install -e .
   pip install -r requirements.txt
   pip install torch torchaudio tensorboardX librosa soundfile scipy
   ```

## Setup

1. Copy TDNN implementation files uploaded:
   ```bash
   cp tdnn.py wenet/transformer/
   cp tdnn_conformer.py wenet/transformer/
   cp tdnn_encoder_layer.py wenet/transformer/
   cp train_tdnn.yaml examples/librispeech/s0/conf/
   ```

2. Update `wenet/transformer/__init__.py`:
   ```python
   from wenet.transformer.tdnn import TDNNModule
   from wenet.transformer.tdnn_conformer import TDNNConformerEncoder
   from wenet.transformer.tdnn_encoder_layer import TDNNConformerEncoderLayer
   ```

3. Modify `wenet/utils/init_model.py`:
   ```python
   WENET_ENCODER_CLASSES = {
       "transformer": TransformerEncoder,
       "conformer": ConformerEncoder,
       "tdnn_conformer": TDNNConformerEncoder,  # Add this line
       # ... other encoders ...
   }
   ```

4. Replace `wenet/examples/librispeech/s0/run.sh` with the uploaded `run.sh` file (keep it in the same exact directory location and do not rename)

5. Prepare data:
   ```bash
   cd examples/librispeech/s0
   mkdir -p data
   
   # Download LibriSpeech
   bash run.sh --stage -1 --stop_stage -1
   
   # Prepare training data
   bash run.sh --stage 0 --stop_stage 0
   
   # Generate features
   bash run.sh --stage 1 --stop_stage 1
   
   # Create dictionary
   bash run.sh --stage 2 --stop_stage 2
   
   # Format data for WeNet
   bash run.sh --stage 3 --stop_stage 3
   ```

## Training

1. Start training:
   ```bash
   bash run.sh --stage 4 --stop_stage 4
   ```

2. Monitor progress:
   ```bash
   tensorboard --logdir tensorboard --port 6006 --bind_all
   ```

## Evaluation

Run evaluation on test sets to create files with Word Error Rate related data:
```bash
bash run.sh --stage 5 --stop_stage 5
```

## Configuration

Key settings in `train_tdnn.yaml`:

```yaml
encoder: tdnn_conformer
encoder_conf:
    output_size: 256
    attention_heads: 4
    linear_units: 1024
    num_blocks: 6
    dropout_rate: 0.1
    
    # TDNN specific
    tdnn_module_kernel: 3
    tdnn_module_dilation: 1
    tdnn_module_context_size: 2
    use_tdnn_norm: "batch_norm"
    causal: false
```

## Directory Structure

```
wenet/
├── transformer/
│   ├── tdnn.py
│   ├── tdnn_conformer.py
│   └── tdnn_encoder_layer.py
└── examples/
    └── librispeech/
        └── s0/
            └── conf/
                └── train_tdnn.yaml
```

### System Requirements

- Training: 16GB+ GPU RAM
- Inference: 8GB+ GPU RAM
- Storage: 50GB+ for data

### Documentation

Refer to WeNet official documentation for further clarification:

1. WeNet Setup Tutorial: https://wenet.org.cn/wenet/python_package.html#install

2. Librispeech Dataset Specific Tutorial: https://wenet.org.cn/wenet/tutorial_librispeech.html
