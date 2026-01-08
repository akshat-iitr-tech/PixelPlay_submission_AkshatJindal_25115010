# 🎥 Video Anomaly Detection on Avenue Dataset

Unsupervised Spatiotemporal Anomaly Detection using I3D

**Author:** Akshat Jindal  
**Challenge:** VLG Recruitment Challenge ’26  
**Repository:** PixelPlay_submission_AkshatJindal_25115010

---

## 📌 Overview

This project presents a high-performance, unsupervised Video Anomaly Detection (VAD) system for surveillance footage using the Avenue Dataset.

The core principle is simple and effective:

Learn normal spatiotemporal behavior from video → detect deviations as anomalies

The system is designed with a strong focus on:

- True motion modeling
- Robust handling of dataset artifacts
- Efficient, competition-ready inference

---

## 🧠 High-Level Pipeline

Video Frames  
↓  
Dataset-Aware Preprocessing  
↓  
16-Frame Clip Construction  
↓  
I3D Feature Extraction (2048-D)  
↓  
Unsupervised Normalcy Modeling (AE / k-NN)  
↓  
Temporal Aggregation & Smoothing  
↓  
Frame-Level Anomaly Score (0–1)

---

## ⚙️ Key Implementation Details (Core Strengths)

### 1️⃣ Dataset-Aware Preprocessing (Critical)

The Avenue dataset contains intentional test-time corruption that can severely degrade VAD performance if ignored.

This implementation explicitly addresses them.

- ✔ Inversion Detection & Correction (Test Set Only)

  Detected using:

  - Brightness comparison (top vs bottom regions)
  - HSV-based grass / pavement heuristics

  Corrected using vertical flipping (cv2.flip)

  📈 ~30% of test frames were auto-corrected, aligning with visual inspection and significantly improving anomaly scores.

- ✔ Noise & Blur Handling

  Fast detection using Laplacian variance

  Correction using:

  - Gaussian blur (fast path)
  - Non-Local Means denoising (selective)

  All corrections are test-only, ensuring the training distribution remains clean.

### 2️⃣ Spatiotemporal Feature Extraction (I3D)

- Backbone: I3D (Inflated 3D ConvNet, ResNet-50 based)  
- Pretrained on: Kinetics-400  
- Input: (B, 3, 16, 224, 224)  
- Output: 2048-D feature vector per clip  
- Usage: Frozen (feature extractor only)

Why I3D?

Uses 3D convolutions → captures motion directly

Learns how things move, not just what they look like

Ideal for anomalies like running, throwing, loitering

### 3️⃣ Normalcy Modeling (Unsupervised)

#### 🔹 Autoencoder-Based Anomaly Scoring (Primary)

- Architecture: Bottleneck MLP Autoencoder

  2048 → 1024 → 512 → 128 → 512 → 1024 → 2048

- Training: Normal videos only  
- Loss: Mean Squared Error (MSE)  
- Regularization: BatchNorm + Dropout + Xavier Init

Rationale:  
The bottleneck forces the model to learn only normal motion patterns.  
Anomalies fail to reconstruct → high reconstruction error.

#### 🔹 FAISS k-NN (Alternative)

Stores all normal I3D features in a FAISS index

Anomaly score = distance to nearest neighbors

Zero training time

Extremely fast inference

| Aspect       | Autoencoder | k-NN       |
|--------------|-------------|------------|
| Training     | Required    | None       |
| Memory       | Constant    | Linear     |
| Overfitting  | Possible    | Very robust|
| Speed        | Fast        | Very fast  |

Both methods achieve similar AUC; AE is slightly higher, k-NN is more stable.

### 4️⃣ Temporal Scoring & Output

- Sliding window with overlap
- Train stride: 2 (efficiency)
- Test stride: 1 (fine localization)
- Frame score = max score from overlapping clips
- Gaussian temporal smoothing
- Global normalization to [0, 1]

✔ Produces frame-level anomaly probability, ready for submission.

---

## 🚀 Performance Optimizations

- Vectorized ImageNet normalization
- Parallel frame loading (ThreadPoolExecutor)
- Persistent feature caching (.pt)
- AMP-enabled training
- Fully vectorized scoring & aggregation

These choices eliminate I/O bottlenecks and enable rapid experimentation.

---

## 🏗️ Training Configuration

| Component       | Value                       |
|-----------------|-----------------------------|
| I3D Batch Size  | 4                           |
| AE Batch Size   | 128                         |
| Optimizer       | AdamW                       |
| Learning Rate   | 5e-4 (Cosine Annealing)     |
| Epochs          | 100                         |
| Hardware        | NVIDIA P100 (Kaggle)        |

---

## 🧪 Additional Experiments

To gain deeper insight, multiple alternative architectures were explored:

- I3D + FAISS k-NN
- VGG + ConvLSTM Autoencoder
- 3D CNN + Attention-based AE (from scratch)
- ConvLSTM spatiotemporal AE

Key observation:

Feature quality matters more than the anomaly scoring method.

---

## 📌 Key Takeaways

- Video anomaly detection is inherently spatiotemporal
- 2D CNNs are insufficient for motion-based anomalies
- Dataset artifacts can destroy AUC if ignored
- Unsupervised methods (AE, k-NN) are highly effective
- Engineering decisions matter as much as model choice

---

## 📎 Repository

- 🔗 GitHub: [https://github.com/akshat-iitr-tech/PixelPlay_submission_AkshatJindal_25115010](https://github.com/akshat-iitr-tech/PixelPlay_submission_AkshatJindal_25115010)
