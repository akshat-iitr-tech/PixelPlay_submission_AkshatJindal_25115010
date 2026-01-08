# PixelPlay_submission_AkshatJindal_25115010
My submission for VLG PixelPlay 26. Akshat Jindal 25115010

🎥 Video Anomaly Detection on Avenue Dataset

Unsupervised Spatiotemporal Anomaly Detection using I3D

Author: Akshat Jindal
Challenge: VLG Recruitment Challenge ’26

📌 Project Summary

This project implements a high-accuracy, unsupervised Video Anomaly Detection (VAD) system for surveillance videos using the Avenue Dataset.

The core idea is to:

Learn normal motion patterns using spatiotemporal features and detect deviations as anomalies.

The system is optimized for robustness, speed, and competition-ready evaluation.

🧠 Core Architecture
1️⃣ Spatiotemporal Feature Extraction (I3D)

Backbone: I3D (Inflated 3D ConvNet, ResNet-50 based)

Pretrained on: Kinetics-400

Input: 16-frame clips (B, 3, 16, 224, 224)

Output: 2048-D feature vector per clip

Usage: Frozen feature extractor (no fine-tuning)

Why I3D?

Captures motion explicitly using 3D convolutions

Ideal for anomalies like running, throwing, loitering

Far superior to 2D CNNs for video anomaly detection

2️⃣ Normalcy Modeling (Unsupervised)
🔹 A. Autoencoder-Based Scoring (Primary)

Architecture: Bottleneck MLP Autoencoder
2048 → 1024 → 512 → 128 → 512 → 1024 → 2048

Training: Only on normal videos

Anomaly Score: Reconstruction MSE

Regularization: BatchNorm + Dropout + Xavier init

Key Insight:
The bottleneck forces the model to learn only normal behavior.
Anomalies fail to reconstruct → high error.

🔹 B. FAISS k-NN (Alternative)

Stores all normal I3D features in a FAISS index

Anomaly = large distance to nearest neighbors

No training required

Extremely fast inference

Both AE and k-NN achieve similar AUC; AE slightly higher, k-NN more robust.

⚙️ Vital Implementation Details (Key Strengths)
🔧 Dataset-Specific Preprocessing (Critical)

The Avenue dataset contains intentional corruption in the test set.

Handled explicitly in this implementation:

Upside-down frames (test-only)

Detected using brightness comparison (top vs bottom)

HSV-based grass & pavement heuristics

Corrected using vertical flip

Noise & blur

Laplacian variance for detection

Fast Gaussian / NLM denoising

📈 ~30% of test frames were auto-corrected, significantly improving performance.

⚡ High-Performance Pipeline

Vectorized ImageNet normalization

Parallel frame loading (ThreadPoolExecutor)

Feature caching (.pt) to decouple I3D from training

AMP-enabled training

Sliding window with stride control

Train stride = 2 (efficiency)

Test stride = 1 (fine localization)

📊 Anomaly Scoring & Output

Reconstruction error → clip-level score

Frame-level score = max over overlapping clips

Temporal Gaussian smoothing

Global normalization to [0, 1]

Outputs per-frame anomaly probability (competition-ready)

🏗️ Training Configuration
Component	Value
I3D Batch Size	4
AE Batch Size	128
Optimizer	AdamW
Learning Rate	5e-4 (Cosine Annealing)
Epochs	100
Hardware	NVIDIA P100 (Kaggle)
📌 Key Takeaways

Video anomaly detection is fundamentally spatiotemporal

Feature quality matters more than scoring method

Dataset artifacts can destroy AUC if ignored

Unsupervised methods (AE / k-NN) are highly effective

Engineering decisions matter as much as model choice

📎 Repository

🔗 GitHub:
https://github.com/akshat-iitr-tech/PixelPlay_submission_AkshatJindal_25115010
