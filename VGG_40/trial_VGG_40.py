# =============================================================================
# AVENUE DATASET ANOMALY DETECTION (Ultra-Fast with VGG Feature Pre-extraction)
# Pre-computes VGG features once, then trains only the lightweight autoencoder
# Expected: ~30-45 seconds per epoch (down from 3 minutes)
# =============================================================================

import os
import random
import time
import glob
import hashlib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import cv2
from tqdm.auto import tqdm
from collections import defaultdict

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, TensorDataset
import torchvision.models as models

# ===============================
# CONFIGURATION
# ===============================

IMG_SIZE = 224
SEQ_LEN = 10
BATCH_SIZE = 16          # Can use larger batch now (no VGG in training loop)
NUM_EPOCHS = 10
LEARNING_RATE = 1e-4     # Slightly higher LR for faster convergence
WEIGHT_DECAY = 1e-5
NUM_WORKERS = 2          # Fewer workers needed for pre-extracted features
FEATURE_CACHE_DIR = "/kaggle/working/vgg_features"

# Paths
DATA_ROOT = "/kaggle/input/avenue-dataset/Avenue_Corrupted/Dataset"
TRAIN_DIR = os.path.join(DATA_ROOT, "training_videos")
TEST_DIR = os.path.join(DATA_ROOT, "testing_videos")
OUTPUT_CSV = "avenue_anomaly_scores.csv"

# Reproducibility
SEED = 42
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(SEED)
    torch.backends.cudnn.deterministic = False
    torch.backends.cudnn.benchmark = True

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

os.makedirs(FEATURE_CACHE_DIR, exist_ok=True)

# ===============================
# VGG FEATURE EXTRACTOR (Pre-extraction only)
# ===============================

class VGGFeatureExtractor(nn.Module):
    """Extracts VGG features. Used only during pre-extraction phase."""
    def __init__(self):
        super().__init__()
        vgg = models.vgg16(weights=models.VGG16_Weights.IMAGENET1K_V1)
        # Use up to pool3: 256 x 28 x 28
        self.features = nn.Sequential(*list(vgg.features[:17]))
        
        for param in self.parameters():
            param.requires_grad = False
        self.eval()
    
    @torch.no_grad()
    def forward(self, x):
        return self.features(x)

# ===============================
# FAST IMAGE LOADING
# ===============================

# ImageNet normalization
MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
STD = np.array([0.229, 0.224, 0.225], dtype=np.float32)

def load_image_fast(path):
    """Load and preprocess image efficiently."""
    img = cv2.imread(path)
    if img is None:
        return np.zeros((3, IMG_SIZE, IMG_SIZE), dtype=np.float32)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = cv2.resize(img, (IMG_SIZE, IMG_SIZE), interpolation=cv2.INTER_LINEAR)
    img = img.astype(np.float32) / 255.0
    img = (img - MEAN) / STD
    return np.transpose(img, (2, 0, 1))

def extract_frame_num(filepath):
    fname = os.path.basename(filepath)
    fname_no_ext = os.path.splitext(fname)[0]
    digits = ''.join(filter(str.isdigit, fname_no_ext))
    return int(digits) if digits else 0

# ===============================
# VGG FEATURE PRE-EXTRACTION
# ===============================

def preextract_vgg_features(data_dir, cache_name):
    """
    Pre-extract VGG features for all frames and save to disk.
    This is done ONCE before training, eliminating VGG from training loop.
    
    Returns: Dictionary mapping frame_path -> feature tensor
    """
    cache_path = os.path.join(FEATURE_CACHE_DIR, f"{cache_name}_features.pt")
    meta_path = os.path.join(FEATURE_CACHE_DIR, f"{cache_name}_meta.pt")
    
    # Check if already cached
    if os.path.exists(cache_path) and os.path.exists(meta_path):
        print(f"Loading pre-extracted features from {cache_path}...")
        features_dict = torch.load(cache_path, weights_only=False)
        metadata = torch.load(meta_path, weights_only=False)
        print(f"Loaded {len(features_dict)} cached features.")
        return features_dict, metadata
    
    print(f"Pre-extracting VGG features for {cache_name}...")
    
    # Initialize VGG
    vgg = VGGFeatureExtractor().to(device)
    
    # Collect all frame paths
    video_folders = sorted(os.listdir(data_dir))
    all_frames = []
    frame_to_video = {}
    
    for vid_idx, vid_name in enumerate(video_folders, start=1):
        vid_path = os.path.join(data_dir, vid_name)
        frames = sorted(glob.glob(os.path.join(vid_path, "*.jpg")))
        for fp in frames:
            all_frames.append(fp)
            frame_to_video[fp] = (vid_idx, extract_frame_num(fp))
    
    print(f"Found {len(all_frames)} frames to process.")
    
    # Extract features in batches
    features_dict = {}
    batch_size = 64  # Large batch for fast extraction
    
    for i in tqdm(range(0, len(all_frames), batch_size), desc="Extracting VGG features"):
        batch_paths = all_frames[i:i + batch_size]
        
        # Load batch
        batch_imgs = np.stack([load_image_fast(p) for p in batch_paths])
        batch_tensor = torch.from_numpy(batch_imgs).to(device)
        
        # Extract features
        with torch.amp.autocast('cuda'):
            batch_features = vgg(batch_tensor)  # (B, 256, 28, 28)
        
        # Store features (move to CPU to save GPU memory)
        batch_features = batch_features.cpu()
        for j, path in enumerate(batch_paths):
            features_dict[path] = batch_features[j]
    
    # Save to disk
    print(f"Saving features to {cache_path}...")
    torch.save(features_dict, cache_path)
    torch.save(frame_to_video, meta_path)
    
    # Clear GPU memory
    del vgg
    torch.cuda.empty_cache()
    
    print(f"Pre-extraction complete. Saved {len(features_dict)} features.")
    return features_dict, frame_to_video

# ===============================
# FAST FEATURE DATASET
# ===============================

class PreextractedFeatureDataset(Dataset):
    """
    Dataset that loads pre-extracted VGG features.
    No VGG computation during training = FAST!
    """
    def __init__(self, data_dir, features_dict, frame_to_video, is_train=True):
        self.features_dict = features_dict
        self.frame_to_video = frame_to_video
        self.is_train = is_train
        self.samples = []
        self.metadata = []
        self.all_video_frames = []
        
        video_folders = sorted(os.listdir(data_dir))
        
        for vid_idx, vid_name in enumerate(video_folders, start=1):
            vid_path = os.path.join(data_dir, vid_name)
            frames = sorted(glob.glob(os.path.join(vid_path, "*.jpg")))
            
            if not frames:
                continue
            
            for fp in frames:
                self.all_video_frames.append((vid_idx, extract_frame_num(fp)))
            
            if len(frames) < SEQ_LEN:
                continue
            
            stride = 2 if is_train else 1
            
            for i in range(0, len(frames) - SEQ_LEN + 1, stride):
                window = frames[i:i + SEQ_LEN]
                # Only add if all frames have features
                if all(p in features_dict for p in window):
                    self.samples.append(window)
                    meta = [(vid_idx, extract_frame_num(f)) for f in window]
                    self.metadata.append(meta)
        
        print(f"{'Train' if is_train else 'Test'} Dataset: {len(self.samples)} sequences")
    
    def __len__(self):
        return len(self.samples)
    
    def __getitem__(self, idx):
        paths = self.samples[idx]
        # Stack pre-extracted features: (T, 256, 28, 28)
        features = torch.stack([self.features_dict[p] for p in paths])
        return features

# ===============================
# LIGHTWEIGHT AUTOENCODER (No VGG)
# ===============================

class ConvLSTMCell(nn.Module):
    def __init__(self, input_dim, hidden_dim, kernel_size):
        super().__init__()
        self.hidden_dim = hidden_dim
        padding = kernel_size // 2
        self.gates = nn.Conv2d(input_dim + hidden_dim, 4 * hidden_dim, 
                               kernel_size, 1, padding, bias=False)
        self.norm = nn.GroupNorm(4, 4 * hidden_dim)
    
    def forward(self, x, state):
        h, c = state
        gates = self.norm(self.gates(torch.cat([x, h], dim=1)))
        i, f, o, g = gates.chunk(4, dim=1)
        i, f, o = torch.sigmoid(i), torch.sigmoid(f), torch.sigmoid(o)
        g = torch.tanh(g)
        c_next = f * c + i * g
        h_next = o * torch.tanh(c_next)
        return h_next, c_next

class BidirectionalConvLSTM(nn.Module):
    def __init__(self, input_dim, hidden_dim, kernel_size):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.fwd_cell = ConvLSTMCell(input_dim, hidden_dim, kernel_size)
        self.bwd_cell = ConvLSTMCell(input_dim, hidden_dim, kernel_size)
        self.fusion = nn.Sequential(
            nn.Conv2d(hidden_dim * 2, hidden_dim, 1, bias=False),
            nn.BatchNorm2d(hidden_dim),
            nn.ReLU(inplace=True)
        )
    
    def forward(self, x):
        B, T, C, H, W = x.size()
        dtype, dev = x.dtype, x.device
        
        h_f = torch.zeros(B, self.hidden_dim, H, W, dtype=dtype, device=dev)
        c_f = torch.zeros_like(h_f)
        h_b = torch.zeros_like(h_f)
        c_b = torch.zeros_like(h_f)
        
        fwd_out, bwd_out = [], [None] * T
        for t in range(T):
            h_f, c_f = self.fwd_cell(x[:, t], (h_f, c_f))
            fwd_out.append(h_f)
        for t in range(T - 1, -1, -1):
            h_b, c_b = self.bwd_cell(x[:, t], (h_b, c_b))
            bwd_out[t] = h_b
        
        fwd_stack = torch.stack(fwd_out, dim=1)
        bwd_stack = torch.stack(bwd_out, dim=1)
        combined = torch.cat([fwd_stack, bwd_stack], dim=2)
        
        B, T, C2, H, W = combined.shape
        fused = self.fusion(combined.view(B * T, C2, H, W))
        return fused.view(B, T, self.hidden_dim, H, W)

class ChannelAttention(nn.Module):
    def __init__(self, channels, reduction=8):
        super().__init__()
        self.fc = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Linear(channels, channels // reduction),
            nn.ReLU(inplace=True),
            nn.Linear(channels // reduction, channels),
            nn.Sigmoid()
        )
    
    def forward(self, x):
        B, T, C, H, W = x.size()
        x_flat = x.view(B * T, C, H, W)
        weights = self.fc(x_flat).view(B * T, C, 1, 1)
        return (x_flat * weights).view(B, T, C, H, W)

class FastFeatureAutoencoder(nn.Module):
    """
    Lightweight autoencoder that works on pre-extracted VGG features.
    Input: (B, T, 256, 28, 28) - pre-extracted VGG features
    No VGG computation here = FAST training!
    """
    def __init__(self):
        super().__init__()
        
        # Encoder: 256 -> 128 -> 64
        self.enc1 = nn.Sequential(
            nn.Conv2d(256, 128, 3, padding=1, bias=False),
            nn.BatchNorm2d(128),
            nn.LeakyReLU(0.2, inplace=True)
        )
        self.enc2 = nn.Sequential(
            nn.Conv2d(128, 64, 3, padding=1, bias=False),
            nn.BatchNorm2d(64),
            nn.LeakyReLU(0.2, inplace=True)
        )
        
        # Temporal modeling
        self.bilstm = BidirectionalConvLSTM(64, 64, 3)
        self.attention = ChannelAttention(64)
        
        # Decoder: 64 -> 128 -> 256
        self.dec2 = nn.Sequential(
            nn.ConvTranspose2d(64, 128, 3, padding=1, bias=False),
            nn.BatchNorm2d(128),
            nn.LeakyReLU(0.2, inplace=True)
        )
        self.dec1 = nn.Sequential(
            nn.ConvTranspose2d(256, 256, 3, padding=1, bias=False),  # 128 + 128 skip
            nn.BatchNorm2d(256)
        )
    
    def forward(self, x):
        """
        x: (B, T, 256, 28, 28) - pre-extracted VGG features
        Returns: input features (target), reconstructed features
        """
        B, T, C, H, W = x.size()
        target = x.view(B * T, C, H, W)  # Original features as target
        
        # Encode
        e1 = self.enc1(target)  # (B*T, 128, 28, 28)
        e2 = self.enc2(e1)      # (B*T, 64, 28, 28)
        
        # Reshape for LSTM
        e2_seq = e2.view(B, T, 64, H, W)
        
        # BiLSTM + Attention
        lstm_out = self.bilstm(e2_seq)      # (B, T, 64, 28, 28)
        lstm_out = self.attention(lstm_out)  # (B, T, 64, 28, 28)
        lstm_out = lstm_out.view(B * T, 64, H, W)
        
        # Decode with skip connection
        d2 = self.dec2(lstm_out)              # (B*T, 128, 28, 28)
        d2 = torch.cat([d2, e1], dim=1)       # (B*T, 256, 28, 28)
        reconstruction = self.dec1(d2)         # (B*T, 256, 28, 28)
        
        return target, reconstruction

# ===============================
# OPTIMIZED LOSS
# ===============================

class CombinedLoss(nn.Module):
    def __init__(self):
        super().__init__()
        self.mse = nn.MSELoss()
    
    def forward(self, pred, target):
        # MSE
        mse_loss = self.mse(pred, target)
        
        # Cosine similarity
        pred_flat = pred.view(pred.size(0), -1)
        target_flat = target.view(target.size(0), -1)
        cosine_loss = 1 - F.cosine_similarity(pred_flat, target_flat, dim=1).mean()
        
        return mse_loss + 0.2 * cosine_loss, mse_loss, cosine_loss

# ===============================
# FAST TRAINING
# ===============================

def train():
    print("=" * 60)
    print("PHASE 1: Pre-extracting VGG Features (one-time)")
    print("=" * 60)
    
    # Pre-extract features (cached to disk)
    train_features, train_meta = preextract_vgg_features(TRAIN_DIR, "train")
    
    print("\n" + "=" * 60)
    print("PHASE 2: Training Autoencoder (FAST - no VGG)")
    print("=" * 60)
    
    # Create dataset with pre-extracted features
    train_ds = PreextractedFeatureDataset(TRAIN_DIR, train_features, train_meta, is_train=True)
    
    train_dl = DataLoader(
        train_ds,
        batch_size=BATCH_SIZE,
        shuffle=True,
        num_workers=NUM_WORKERS,
        pin_memory=True,
        persistent_workers=True if NUM_WORKERS > 0 else False,
        drop_last=True
    )
    
    # Model (lightweight, no VGG)
    model = FastFeatureAutoencoder().to(device)
    
    params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Trainable parameters: {params:,} (VGG excluded)")
    
    optimizer = optim.AdamW(model.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY)
    scheduler = optim.lr_scheduler.OneCycleLR(
        optimizer,
        max_lr=LEARNING_RATE * 10,
        epochs=NUM_EPOCHS,
        steps_per_epoch=len(train_dl),
        pct_start=0.1
    )
    
    criterion = CombinedLoss()
    scaler = torch.amp.GradScaler('cuda')
    
    print(f"\nStarting training ({NUM_EPOCHS} epochs, ~{len(train_dl)} batches/epoch)")
    best_loss = float('inf')
    
    for epoch in range(NUM_EPOCHS):
        model.train()
        epoch_loss = 0
        epoch_start = time.time()
        
        pbar = tqdm(train_dl, desc=f"Epoch {epoch+1}/{NUM_EPOCHS}", leave=False)
        
        for batch in pbar:
            batch = batch.to(device, non_blocking=True)
            
            optimizer.zero_grad(set_to_none=True)
            
            with torch.amp.autocast('cuda'):
                target, recon = model(batch)
                loss, mse, cos = criterion(recon, target)
            
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            scaler.step(optimizer)
            scaler.update()
            scheduler.step()
            
            epoch_loss += loss.item()
            pbar.set_postfix({'loss': f'{loss.item():.4f}'})
        
        epoch_time = time.time() - epoch_start
        avg_loss = epoch_loss / len(train_dl)
        
        print(f"Epoch {epoch+1:2d} | Loss: {avg_loss:.6f} | Time: {epoch_time:.1f}s | "
              f"LR: {scheduler.get_last_lr()[0]:.2e}")
        
        if avg_loss < best_loss:
            best_loss = avg_loss
            torch.save(model.state_dict(), 'best_model.pth')
    
    model.load_state_dict(torch.load('best_model.pth', weights_only=True))
    print(f"\nTraining complete. Best loss: {best_loss:.6f}")
    
    return model, train_features

# ===============================
# EVALUATION
# ===============================

def evaluate(model):
    print("\n" + "=" * 60)
    print("PHASE 3: Evaluation")
    print("=" * 60)
    
    # Pre-extract test features
    test_features, test_meta = preextract_vgg_features(TEST_DIR, "test")
    
    test_ds = PreextractedFeatureDataset(TEST_DIR, test_features, test_meta, is_train=False)
    test_dl = DataLoader(
        test_ds,
        batch_size=BATCH_SIZE * 2,
        shuffle=False,
        num_workers=NUM_WORKERS,
        pin_memory=True
    )
    
    model.eval()
    all_errors = []
    all_metadata = []
    
    print("Computing reconstruction errors...")
    with torch.no_grad():
        for i, batch in enumerate(tqdm(test_dl, desc="Evaluating")):
            batch = batch.to(device, non_blocking=True)
            B_actual = batch.size(0)
            
            with torch.amp.autocast('cuda'):
                target, recon = model(batch)
            
            # MSE per frame
            error = ((target - recon) ** 2).mean(dim=(1, 2, 3))
            error = error.view(B_actual, SEQ_LEN)
            all_errors.append(error.cpu())
            
            start_idx = i * (BATCH_SIZE * 2)
            for b in range(B_actual):
                if start_idx + b < len(test_ds.metadata):
                    all_metadata.append(test_ds.metadata[start_idx + b])
    
    # Aggregate scores (convert to float32 for scipy compatibility)
    all_errors = torch.cat([e.view(-1) for e in all_errors], dim=0).float().numpy()
    
    frame_scores = defaultdict(list)
    idx = 0
    for seq_meta in all_metadata:
        for vid_id, frame_num in seq_meta:
            if idx < len(all_errors):
                frame_scores[(vid_id, frame_num)].append(all_errors[idx])
            idx += 1
    
    all_frames = sorted(list(set(test_ds.all_video_frames)))
    raw_scores = np.array([
        np.mean(frame_scores.get((v, f), [0.0])) for v, f in all_frames
    ], dtype=np.float32)  # Ensure float32
    ids = [f"{v}_{f}" for v, f in all_frames]
    
    # Temporal smoothing
    try:
        from scipy.ndimage import gaussian_filter1d
        scores_smooth = gaussian_filter1d(raw_scores.astype(np.float64), sigma=3)
    except ImportError:
        kernel = np.ones(7) / 7
        scores_smooth = np.convolve(raw_scores, kernel, mode='same')
    
    # Normalize
    p01, p99 = np.percentile(scores_smooth, [1, 99])
    scores_clip = np.clip(scores_smooth, p01, p99)
    scores_norm = (scores_clip - p01) / (p99 - p01 + 1e-8)
    
    threshold = np.mean(scores_norm) + 2 * np.std(scores_norm)
    print(f"Threshold: {threshold:.4f}, Anomalies: {np.sum(scores_norm > threshold)}/{len(scores_norm)}")
    
    # Save
    df = pd.DataFrame({'Id': ids, 'Score': scores_norm})
    df.to_csv(OUTPUT_CSV, index=False)
    print(f"Saved to {OUTPUT_CSV}")
    
    # Plot
    fig, axes = plt.subplots(2, 1, figsize=(15, 8))
    
    axes[0].plot(scores_norm, color='crimson', linewidth=0.7)
    axes[0].axhline(threshold, color='orange', linestyle='--', label=f'Threshold')
    axes[0].fill_between(range(len(scores_norm)), scores_norm, threshold,
                          where=scores_norm > threshold, alpha=0.3, color='red')
    axes[0].set_title("Anomaly Scores")
    axes[0].set_xlabel("Frame")
    axes[0].set_ylabel("Score")
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)
    
    axes[1].hist(scores_norm, bins=50, color='steelblue', edgecolor='black', alpha=0.7)
    axes[1].axvline(threshold, color='red', linestyle='--')
    axes[1].set_title("Score Distribution")
    axes[1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('anomaly_analysis.png', dpi=150)
    plt.show()
    
    return scores_norm, threshold

# ===============================
# MAIN
# ===============================

if __name__ == "__main__":
    total_start = time.time()
    
    # Train
    model, _ = train()
    
    # Evaluate
    scores, threshold = evaluate(model)
    
    total_time = time.time() - total_start
    print(f"\n{'='*60}")
    print(f"Total time: {total_time/60:.1f} minutes")
    print(f"{'='*60}")
