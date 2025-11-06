# 🌀 CycleGAN: Unpaired Image-to-Image Translation

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/Joseph1997-eng/CycleGAN/blob/main/C3W3_Assignment.ipynb)


This project implements a **CycleGAN (Cycle-Consistent Adversarial Network)** for unpaired image-to-image translation, based on the paper  
[*Unpaired Image-to-Image Translation using Cycle-Consistent Adversarial Networks*](https://arxiv.org/abs/1703.10593) by Zhu et al., 2017.

CycleGAN enables transformation between two visual domains **without requiring paired training data** — for example, translating **horses ↔ zebras**.

---

## 1. 🎯 Project Goals
1. Implement and understand the **CycleGAN architecture**.
2. Build the **Generator** and **Discriminator** networks from scratch using PyTorch.
3. Implement key **loss functions**:
   - Adversarial Loss  
   - Identity Loss  
   - Cycle Consistency Loss
4. Train a model that can translate **Horse ↔ Zebra** images.

---

## 2. 📘 Learning Objectives
- Understand the concept of **unpaired image translation**.  
- Learn about **Cycle Consistency Loss** and **Identity Loss** in GANs.  
- Explore the structure of **Residual Blocks** in generators.  
- Implement a **PatchGAN discriminator**.  

---

## 3. ⚙️ Getting Started

### 3.1. Requirements
Make sure you have the following Python libraries installed:


```bash
pip install torch torchvision matplotlib pillow tqdm
```

### 3.2. Dataset

Download the Horse ↔ Zebra dataset from [*CycleGAN Dataset (horse2zebra)*](https://www.kaggle.com/datasets/suyashdamle/cyclegan) or the official [*TensorFlow dataset*](https://www.tensorflow.org/datasets/catalog/cycle_gan).
Directory Structure:

```bash
horse2zebra/
 ├── trainA/   # Horses
 ├── trainB/   # Zebras
 ├── testA/
 └── testB/
```
---
## 4. 🧠 Model Architecture
### 4.1. Generator

The generator is a U-Net-like encoder-decoder with:

- 2 Contracting Blocks

- 9 Residual Blocks

- 2 Expanding Blocks

- Uses Instance Normalization instead of BatchNorm.

Residual Blocks help retain image details and stabilize deep training.

### 4.2. Discriminator

The discriminator is a PatchGAN:

- Classifies 70×70 image patches as real or fake.

- Encourages high-frequency realism.
---
## 5. Pretrained Model

To accelerate convergence, you can load a pre-trained checkpoint:

```Python
pre_dict = torch.load('cycleGAN_100000.pth')
gen_AB.load_state_dict(pre_dict['gen_AB'])
gen_BA.load_state_dict(pre_dict['gen_BA'])
```
---
## 6. Sample Visualization
```python
def show_tensor_images(image_tensor, num_images=25, size=(3, 256, 256)):
    image_tensor = (image_tensor + 1) / 2
    image_grid = make_grid(image_tensor[:num_images], nrow=5)
    plt.imshow(image_grid.permute(1, 2, 0).squeeze())
    plt.show()
```
---
## 7. Example Usage
```python
# Training Example
python train.py --dataset horse2zebra --epochs 200 --batch_size 1

# Testing Example
python test.py --dataset horse2zebra --checkpoint ./checkpoints/cycleGAN_100000.pth
```
---
## 8. Repository Structure
```bash
CycleGAN/
 ├── datasets/
 │    └── horse2zebra/
 ├── models/
 │    ├── generator.py
 │    ├── discriminator.py
 │    └── utils.py
 ├── outputs/
 │    ├── generated_samples/
 │    └── checkpoints/
 ├── train.py
 ├── test.py
 ├── requirements.txt
 └── README.md
```

## 9. 📥 Download Pretrained Model  
You can download the pretrained checkpoint here:  
[cycleGAN_100000.pth (Google Drive)](https://drive.google.com/file/d/1OxzDsf0Sl5XxCq1M58mi80q3I3iZ25Tb/view?usp=sharing)

