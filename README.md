# Deformation-Aware and Reconstruction-Driven Multimodal Representation Learning for Brain Tumor Segmentation with Missing Modalities


## Environment

- Python 3.9
- Torch 1.12.1
- cuda_11.6

## Data Processing Steps

### Step 1: Data Preprocessing
  ```bash
    python preprocess.py
```

### Step 2: Compile DCN
  ```bash
  cd dcn
  bash make.sh
```

## Train and Test
```bash
python train.py
python test.py
```

## Model Weights
We have retrained our model on an NVIDIA RTX 4090. You can download it if needed.

Weight: [Weight-BraTS2020](https://drive.google.com/file/d/17sMQKkh7JBhPiNAzRe6roGhPoyZVn6-J/view?usp=drive_link)

Log: [Log-BraTS2020](https://drive.google.com/file/d/1nxxBknNQlGd4FdZE7GYKGfYwqhcrJhDf/view?usp=drive_link)

## Installation
1. Clone the repository:
   ```bash
   git clone https://github.com/linda0227/SRMNet.git
   ```
   
   
## Acknowledgments
We would like to extend our gratitude to the following project:
- [D3Dnet](https://github.com/XinyiYing/D3Dnet)


