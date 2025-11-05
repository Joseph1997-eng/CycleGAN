# 🌀 CycleGAN: Unpaired Image-to-Image Translation

This project implements a **CycleGAN (Cycle-Consistent Adversarial Network)** for unpaired image-to-image translation, based on the paper  
[*Unpaired Image-to-Image Translation using Cycle-Consistent Adversarial Networks*](https://arxiv.org/abs/1703.10593) by Zhu et al., 2017.

CycleGAN enables transformation between two visual domains **without requiring paired training data** — for example, translating **horses ↔ zebras**.

---

## 🎯 Project Goals
1. Implement and understand the **CycleGAN architecture**.
2. Build the **Generator** and **Discriminator** networks from scratch using PyTorch.
3. Implement key **loss functions**:
   - Adversarial Loss  
   - Identity Loss  
   - Cycle Consistency Loss
4. Train a model that can translate **Horse ↔ Zebra** images.

---

## 📘 Learning Objectives
- Understand the concept of **unpaired image translation**.  
- Learn about **Cycle Consistency Loss** and **Identity Loss** in GANs.  
- Explore the structure of **Residual Blocks** in generators.  
- Implement a **PatchGAN discriminator**.  

---

## ⚙️ Getting Started

### 1. Requirements
Make sure you have the following Python libraries installed:


```bash
pip install torch torchvision matplotlib pillow tqdm
```

### 2. Dataset

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
# 🧠 Model Architecture
## Generator

The generator is a U-Net-like encoder-decoder with:

- 2 Contracting Blocks

- 9 Residual Blocks

- 2 Expanding Blocks

- Uses Instance Normalization instead of BatchNorm.

Residual Blocks help retain image details and stabilize deep training.

## Discriminator

The discriminator is a PatchGAN:

- Classifies 70×70 image patches as real or fake.

- Encourages high-frequency realism.
---
### 1. Pretrained Model

To accelerate convergence, you can load a pre-trained checkpoint:

```Python
pre_dict = torch.load('cycleGAN_100000.pth')
gen_AB.load_state_dict(pre_dict['gen_AB'])
gen_BA.load_state_dict(pre_dict['gen_BA'])
```
---
### 2. Sample Visualization
```python
def show_tensor_images(image_tensor, num_images=25, size=(3, 256, 256)):
    image_tensor = (image_tensor + 1) / 2
    image_grid = make_grid(image_tensor[:num_images], nrow=5)
    plt.imshow(image_grid.permute(1, 2, 0).squeeze())
    plt.show()
```


