# Self-Supervised Learning Framework

This project implements a Self-Supervised Learning (SSL) framework using the CIFAR-10 dataset and a ResNet-18 backbone. The goal of the project is to train a model to learn robust image representations without relying on labeled data. This framework utilizes contrastive learning with data augmentations and a custom contrastive loss function.

---

## **Features**
**Data Augmentation**:

- Random cropping, flipping, color jitter, grayscale conversion, Gaussian blur, and normalization.
  
**Backbone Architecture**:
  
- ResNet-18 with a custom projection head.
  
**Contrastive Learning**:
  
- Contrastive loss function with positive and negative pair sampling.

**Optimization**:
  
- Gradient clipping and weight decay for numerical stability.
  
**Model Checkpointing**:

 - Save model weights at the end of each epoch.


## **How It Works**
1. **Data Augmentation**:
   - Two augmented views of each image are created for contrastive learning.

2. **Contrastive Loss**:
   - Positive pairs: Augmented views of the same image.
   - Negative pairs: Augmented views of different images.
   - Loss is computed using the similarity of positive pairs while minimizing similarity with negative pairs.

3. **Optimization**:
   - The model uses the Adam optimizer with a learning rate of `3e-4` and weight decay of `1e-4`.
   - Gradient clipping ensures numerical stability.

---

## **Results and Evaluation**
- **Training Loss**:
  - Observe the training loss decreasing across epochs, indicating successful representation learning.
- **Downstream Tasks**:
  - Evaluate the learned embeddings on classification or clustering tasks.

---

## **Future Work**
- Extend the framework to larger datasets like CIFAR-100 or ImageNet.
- Experiment with different backbone architectures (e.g., ResNet-50, ViT).
- Implement multi-GPU training for scalability.
- Add downstream task evaluation directly within the framework.

---

## **Acknowledgments**
- CIFAR-10 dataset: https://www.cs.toronto.edu/~kriz/cifar.html
- PyTorch: https://pytorch.org/
- ResNet-18 architecture.

---
