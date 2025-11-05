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
horse2zebra/
 ├── trainA/   # Horses
 ├── trainB/   # Zebras
 ├── testA/
 └── testB/
