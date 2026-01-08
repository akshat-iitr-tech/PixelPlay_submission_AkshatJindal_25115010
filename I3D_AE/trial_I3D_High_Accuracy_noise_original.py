# =============================================================================
# AVENUE ANOMALY DETECTION - VECTORIZED HIGH ACCURACY VERSION
# I3D + Simple Autoencoder + Feature Normalization + Full Vectorization
# =============================================================================

import os
import random
import time
import glob
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import cv2
from tqdm.auto import tqdm
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor
from typing import List, Dict, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, TensorDataset

# ===============================
# CONFIGURATION
# ===============================

IMG_SIZE = 224
SEQ_LEN = 16
BATCH_SIZE = 4              # For I3D extraction
TRAIN_BATCH = 128           # For autoencoder
NUM_EPOCHS = 100
LEARNING_RATE = 5e-4
WEIGHT_DECAY = 1e-5
NUM_WORKERS = 4

# Cache
CACHE_DIR = "/kaggle/working/cache"
FEATURE_CACHE = os.path.join(CACHE_DIR, "features")

# Heatmap Visualization
ENABLE_HEATMAPS = True        # Toggle to enable/disable heatmap generation
HEATMAP_FRAMES = 8            # Number of frames to visualize
HEATMAP_VIDEO_IDX = 1         # Which test video to visualize (1-indexed)

# t-SNE Visualization
ENABLE_TSNE = True            # Toggle to enable/disable t-SNE visualization
TSNE_PERPLEXITY = 30          # t-SNE perplexity parameter
TSNE_MAX_SAMPLES = 2000       # Max samples for t-SNE (for speed)

# Paths
DATA_ROOT = "/kaggle/input/avenue-dataset/Avenue_Corrupted/Dataset"
TRAIN_DIR = os.path.join(DATA_ROOT, "training_videos")
TEST_DIR = os.path.join(DATA_ROOT, "testing_videos")
OUTPUT_CSV = "avenue_scores.csv"

# Seed
SEED = 42
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(SEED)
torch.backends.cudnn.benchmark = True

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Device: {device}")

os.makedirs(FEATURE_CACHE, exist_ok=True)


# ===============================
# VECTORIZED IMAGE LOADING
# ===============================

# Pre-shaped for broadcasting: (1, 1, 1, 3)
MEAN = np.array([0.45, 0.45, 0.45], dtype=np.float32).reshape(1, 1, 1, 3)
STD = np.array([0.225, 0.225, 0.225], dtype=np.float32).reshape(1, 1, 1, 3)


def load_single_image_raw(path: str) -> np.ndarray:
    """Load single image as uint8 (no normalization yet)."""
    img = cv2.imread(path)
    if img is None:
        return np.zeros((IMG_SIZE, IMG_SIZE, 3), dtype=np.uint8)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = cv2.resize(img, (IMG_SIZE, IMG_SIZE), interpolation=cv2.INTER_LINEAR)
    return img


def load_images_vectorized(paths: List[str]) -> np.ndarray:
    """
    VECTORIZED: Load multiple images in parallel.
    Returns: (N, H, W, C) float32 normalized
    """
    # Parallel I/O loading
    with ThreadPoolExecutor(max_workers=NUM_WORKERS) as executor:
        images = list(executor.map(load_single_image_raw, paths))
    
    # VECTORIZED: Stack and normalize in one operation
    batch = np.stack(images, axis=0).astype(np.float32)  # (N, H, W, C)
    batch = (batch / 255.0 - MEAN) / STD  # Vectorized normalization
    
    return batch


def load_clip_vectorized(frame_paths: List[str]) -> np.ndarray:
    """Load video clip with vectorized normalization."""
    return load_images_vectorized(frame_paths)  # (T, H, W, C)


def get_frame_num(path: str) -> int:
    """Extract frame number from filename."""
    name = os.path.basename(path)
    digits = ''.join(filter(str.isdigit, os.path.splitext(name)[0]))
    return int(digits) if digits else 0


# ===============================
# I3D EXTRACTOR (Singleton)
# ===============================

class I3D(nn.Module):
    """Singleton I3D model - loaded only once."""
    _instance = None
    
    def __new__(cls):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
            cls._instance._initialized = False
        return cls._instance
    
    def __init__(self):
        if self._initialized:
            return
        super().__init__()
        
        print("Loading I3D model...")
        self.model = torch.hub.load('facebookresearch/pytorchvideo', 'i3d_r50', pretrained=True)
        self.model.blocks[-1].proj = nn.Identity()
        
        for p in self.parameters():
            p.requires_grad = False
        self.eval()
        self._initialized = True
    
    @torch.no_grad()
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.model(x)


# ===============================
# VECTORIZED FEATURE EXTRACTION
# ===============================

def extract_features_vectorized(data_dir: str, name: str, stride: int = 1) -> Tuple:
    """
    VECTORIZED feature extraction with caching and proper normalization.
    
    Returns:
        features: (N, 2048) normalized tensor
        clips: List of clip metadata
        videos: Dict of video metadata
        stats: Normalization statistics
    """
    cache_file = os.path.join(FEATURE_CACHE, f"{name}_v2_s{stride}.pt")
    
    if os.path.exists(cache_file):
        print(f"[CACHE HIT] Loading {name}")
        data = torch.load(cache_file, weights_only=False)
        return data['features'], data['clips'], data['videos'], data['stats']
    
    print(f"\n{'='*50}")
    print(f"Extracting features: {name} (stride={stride})")
    print(f"{'='*50}")
    
    i3d = I3D().to(device)
    
    # Collect video info
    videos = {}
    clips = []
    
    video_dirs = sorted([d for d in os.listdir(data_dir) 
                        if os.path.isdir(os.path.join(data_dir, d))])
    
    print(f"Found {len(video_dirs)} videos")
    
    for vid_id, vid_name in enumerate(video_dirs, 1):
        vid_path = os.path.join(data_dir, vid_name)
        frames = sorted(glob.glob(os.path.join(vid_path, "*.jpg")))
        
        if len(frames) < SEQ_LEN:
            continue
        
        # VECTORIZED: Extract all frame numbers at once
        frame_nums = np.array([get_frame_num(f) for f in frames])
        
        videos[vid_id] = {
            'name': vid_name,
            'frames': frames,
            'nums': frame_nums,
            'n': len(frames)
        }
        
        # Create clips
        n_clips_vid = (len(frames) - SEQ_LEN) // stride + 1
        for i in range(n_clips_vid):
            start = i * stride
            clips.append({
                'vid': vid_id,
                'start': start,
                'paths': frames[start:start + SEQ_LEN],
                'nums': frame_nums[start:start + SEQ_LEN]
            })
    
    n_clips = len(clips)
    print(f"Total clips: {n_clips}")
    
    # VECTORIZED: Pre-allocate feature tensor
    features = torch.zeros(n_clips, 2048, dtype=torch.float32)
    
    # Extract features in batches
    for i in tqdm(range(0, n_clips, BATCH_SIZE), desc="Extracting I3D"):
        batch_clips = clips[i:i + BATCH_SIZE]
        batch_size = len(batch_clips)
        
        # VECTORIZED: Load all clips for this batch
        # Each clip: (T, H, W, C)
        batch_data = []
        for clip in batch_clips:
            clip_frames = load_clip_vectorized(clip['paths'])
            batch_data.append(clip_frames)
        
        # Stack: (B, T, H, W, C)
        batch_array = np.stack(batch_data, axis=0)
        
        # VECTORIZED: Convert to PyTorch and permute
        # (B, T, H, W, C) -> (B, C, T, H, W)
        batch_tensor = torch.from_numpy(batch_array).permute(0, 4, 1, 2, 3)
        batch_tensor = batch_tensor.to(device)
        
        # Extract with mixed precision
        with torch.amp.autocast('cuda'):
            feats = i3d(batch_tensor)
        
        # Store in pre-allocated tensor
        features[i:i + batch_size] = feats.cpu()
    
    # VECTORIZED: Compute normalization statistics
    stats = {
        'mean': features.mean(dim=0),  # (2048,)
        'std': features.std(dim=0) + 1e-8  # (2048,)
    }
    
    # VECTORIZED: Normalize all features at once
    features = (features - stats['mean'].unsqueeze(0)) / stats['std'].unsqueeze(0)
    
    print(f"Feature stats after normalization: mean={features.mean():.4f}, std={features.std():.4f}")
    
    # Save cache
    torch.save({
        'features': features,
        'clips': clips,
        'videos': videos,
        'stats': stats
    }, cache_file)
    
    del i3d
    torch.cuda.empty_cache()
    
    return features, clips, videos, stats


def extract_test_features_vectorized(data_dir: str, train_stats: dict, stride: int = 1) -> Tuple:
    """
    VECTORIZED test feature extraction using TRAINING normalization stats.
    """
    cache_file = os.path.join(FEATURE_CACHE, f"test_v2_s{stride}.pt")
    
    if os.path.exists(cache_file):
        print(f"[CACHE HIT] Loading test features")
        data = torch.load(cache_file, weights_only=False)
        return data['features'], data['clips'], data['videos']
    
    print(f"\n{'='*50}")
    print(f"Extracting test features (stride={stride})")
    print(f"{'='*50}")
    
    i3d = I3D().to(device)
    
    videos = {}
    clips = []
    
    video_dirs = sorted([d for d in os.listdir(data_dir) 
                        if os.path.isdir(os.path.join(data_dir, d))])
    
    print(f"Found {len(video_dirs)} videos")
    
    for vid_id, vid_name in enumerate(video_dirs, 1):
        vid_path = os.path.join(data_dir, vid_name)
        frames = sorted(glob.glob(os.path.join(vid_path, "*.jpg")))
        
        if len(frames) < SEQ_LEN:
            continue
        
        frame_nums = np.array([get_frame_num(f) for f in frames])
        
        videos[vid_id] = {
            'name': vid_name,
            'frames': frames,
            'nums': frame_nums,
            'n': len(frames)
        }
        
        n_clips_vid = (len(frames) - SEQ_LEN) // stride + 1
        for i in range(n_clips_vid):
            start = i * stride
            clips.append({
                'vid': vid_id,
                'start': start,
                'paths': frames[start:start + SEQ_LEN],
                'nums': frame_nums[start:start + SEQ_LEN]
            })
    
    n_clips = len(clips)
    print(f"Total clips: {n_clips}")
    
    # Pre-allocate
    features = torch.zeros(n_clips, 2048, dtype=torch.float32)
    
    for i in tqdm(range(0, n_clips, BATCH_SIZE), desc="Extracting I3D"):
        batch_clips = clips[i:i + BATCH_SIZE]
        batch_size = len(batch_clips)
        
        batch_data = [load_clip_vectorized(c['paths']) for c in batch_clips]
        batch_array = np.stack(batch_data, axis=0)
        batch_tensor = torch.from_numpy(batch_array).permute(0, 4, 1, 2, 3).to(device)
        
        with torch.amp.autocast('cuda'):
            feats = i3d(batch_tensor)
        
        features[i:i + batch_size] = feats.cpu()
    
    # VECTORIZED: Normalize using TRAINING statistics
    features = (features - train_stats['mean'].unsqueeze(0)) / train_stats['std'].unsqueeze(0)
    
    print(f"Feature stats: mean={features.mean():.4f}, std={features.std():.4f}")
    
    torch.save({
        'features': features,
        'clips': clips,
        'videos': videos
    }, cache_file)
    
    del i3d
    torch.cuda.empty_cache()
    
    return features, clips, videos


# ===============================
# SIMPLE VECTORIZED AUTOENCODER
# ===============================

class SimpleAutoencoder(nn.Module):
    """Simple autoencoder with vectorized operations."""
    
    def __init__(self, input_dim: int = 2048, hidden_dim: int = 1024, latent_dim: int = 128):
        super().__init__()
        
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(0.2),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.BatchNorm1d(hidden_dim // 2),
            nn.ReLU(inplace=True),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim // 2, latent_dim),
        )
        
        self.decoder = nn.Sequential(
            nn.Linear(latent_dim, hidden_dim // 2),
            nn.BatchNorm1d(hidden_dim // 2),
            nn.ReLU(inplace=True),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim // 2, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(0.2),
            nn.Linear(hidden_dim, input_dim),
        )
        
        self._init_weights()
    
    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Fully vectorized forward pass."""
        return self.decoder(self.encoder(x))
    
    def compute_loss(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """VECTORIZED: Compute reconstruction and loss."""
        recon = self.forward(x)
        loss = F.mse_loss(recon, x)
        return recon, loss
    
    @torch.no_grad()
    def compute_errors_vectorized(self, x: torch.Tensor) -> torch.Tensor:
        """VECTORIZED: Compute per-sample MSE errors."""
        recon = self.forward(x)
        # Vectorized MSE per sample: mean over feature dimension
        errors = ((x - recon) ** 2).mean(dim=1)  # (B,)
        return errors


# ===============================
# PIXEL-LEVEL HEATMAP VISUALIZATION
# ===============================

class GradCAMI3D:
    """Grad-CAM for I3D to generate spatial anomaly heatmaps."""
    
    def __init__(self, i3d_model):
        self.model = i3d_model.model
        self.activations = None
        self.gradients = None
        self._register_hooks()
    
    def _register_hooks(self):
        """Register forward and backward hooks on the last conv layer."""
        def forward_hook(module, input, output):
            # Store activations (keep reference for gradient computation)
            self.activations = output
        
        def backward_hook(module, grad_input, grad_output):
            # Store gradients
            self.gradients = grad_output[0]
        
        # Hook into last conv layer (blocks[5] for I3D R50, before pooling)
        self.model.blocks[5].register_forward_hook(forward_hook)
        self.model.blocks[5].register_full_backward_hook(backward_hook)
    
    def compute_heatmap(self, clip_tensor, autoencoder, train_stats):
        """
        Compute Grad-CAM heatmap for a clip.
        
        Args:
            clip_tensor: (1, C, T, H, W) input tensor
            autoencoder: Trained autoencoder model
            train_stats: Training normalization statistics
            
        Returns:
            cam: (T, H, W) heatmap for each frame in clip
            score: Anomaly score for the clip
        """
        autoencoder.eval()
        
        # Reset stored activations/gradients
        self.activations = None
        self.gradients = None
        
        # Enable gradients for this forward pass
        clip_tensor = clip_tensor.clone().requires_grad_(True)
        
        # Forward through I3D model - hooks will capture activations at blocks[5]
        # Temporarily enable gradients for the model
        for param in self.model.parameters():
            param.requires_grad_(True)
        
        features = self.model(clip_tensor)  # (1, 2048)
        
        # Normalize using training stats
        mean = train_stats['mean'].to(features.device)
        std = train_stats['std'].to(features.device)
        features_norm = (features - mean) / std
        
        # Get reconstruction error from autoencoder
        recon = autoencoder(features_norm)
        error = ((features_norm - recon) ** 2).mean()
        score = error.item()
        
        # Backward pass to get gradients
        error.backward()
        
        # Disable gradients again
        for param in self.model.parameters():
            param.requires_grad_(False)
        
        # Get activations and gradients from hooks (detach for numpy conversion)
        activations = self.activations.detach() if self.activations is not None else None
        gradients = self.gradients.detach() if self.gradients is not None else None
        
        if activations is None:
            # Fallback: return uniform heatmap
            print("Warning: No activations captured, using uniform heatmap")
            cam = np.ones((SEQ_LEN, 7, 7), dtype=np.float32)
        elif gradients is None:
            # Fallback: use activation magnitudes if gradients unavailable
            cam = activations.abs().mean(dim=1).squeeze(0).cpu().numpy()  # (T, H', W')
        else:
            # Grad-CAM: weight activations by average gradients
            weights = gradients.mean(dim=(3, 4), keepdim=True)  # (1, C, T, 1, 1)
            cam = F.relu((weights * activations).sum(dim=1)).squeeze(0).cpu().numpy()  # (T, H', W')
        
        return cam, score


def visualize_anomaly_heatmaps(model, train_stats, test_dir, 
                                output_path="pixel_heatmaps.png",
                                n_frames=8, video_idx=1):
    """
    Generate pixel-level anomaly heatmaps for selected frames.
    
    Args:
        model: Trained autoencoder
        train_stats: Training normalization statistics  
        test_dir: Path to test videos
        output_path: Output image path
        n_frames: Number of frames to visualize
        video_idx: Which test video to visualize (1-indexed)
    """
    print(f"\n{'='*50}")
    print(f"Generating Pixel-Level Heatmaps (Video {video_idx})")
    print(f"{'='*50}")
    
    # Load I3D for Grad-CAM
    i3d = I3D().to(device)
    gradcam = GradCAMI3D(i3d)
    
    # Get video frames
    video_dirs = sorted([d for d in os.listdir(test_dir) 
                        if os.path.isdir(os.path.join(test_dir, d))])
    
    if video_idx > len(video_dirs):
        print(f"Video {video_idx} not found! Available: 1-{len(video_dirs)}")
        return None, None
    
    vid_name = video_dirs[video_idx - 1]
    vid_path = os.path.join(test_dir, vid_name)
    frames = sorted(glob.glob(os.path.join(vid_path, "*.jpg")))
    
    if len(frames) < SEQ_LEN:
        print(f"Not enough frames in video {vid_name}")
        return None, None
    
    print(f"Video: {vid_name} ({len(frames)} frames)")
    
    # Select evenly spaced frames
    n_total = len(frames)
    indices = np.linspace(0, n_total - SEQ_LEN, n_frames, dtype=int)
    
    # Storage for visualization
    orig_frames = []
    heatmaps = []
    scores = []
    frame_nums = []
    
    model.eval()
    
    for idx in tqdm(indices, desc="Computing heatmaps"):
        # Load clip
        clip_paths = frames[idx:idx + SEQ_LEN]
        clip_data = load_clip_vectorized(clip_paths)  # (T, H, W, C)
        
        # Get center frame for display
        center_idx = SEQ_LEN // 2
        center_frame_path = clip_paths[center_idx]
        orig_img = cv2.imread(center_frame_path)
        orig_img = cv2.cvtColor(orig_img, cv2.COLOR_BGR2RGB)
        orig_img = cv2.resize(orig_img, (IMG_SIZE, IMG_SIZE))
        orig_frames.append(orig_img)
        frame_nums.append(idx)
        
        # Prepare tensor: (1, C, T, H, W)
        clip_tensor = torch.from_numpy(clip_data).unsqueeze(0).permute(0, 4, 1, 2, 3)
        clip_tensor = clip_tensor.to(device)
        
        # Compute Grad-CAM heatmap
        with torch.enable_grad():
            cam, score = gradcam.compute_heatmap(clip_tensor, model, train_stats)
        
        # cam shape: (T, H', W') - take center frame
        cam_frame = cam[center_idx]
        
        # Resize heatmap to original image size
        cam_resized = cv2.resize(cam_frame, (IMG_SIZE, IMG_SIZE), interpolation=cv2.INTER_LINEAR)
        
        # Normalize heatmap to [0, 1]
        if cam_resized.max() > cam_resized.min():
            cam_resized = (cam_resized - cam_resized.min()) / (cam_resized.max() - cam_resized.min())
        
        heatmaps.append(cam_resized)
        scores.append(score)
    
    # Normalize scores across all frames for display
    scores = np.array(scores)
    if scores.max() > scores.min():
        scores_norm = (scores - scores.min()) / (scores.max() - scores.min())
    else:
        scores_norm = scores
    
    # Create visualization (matching example format)
    fig, axes = plt.subplots(2, n_frames, figsize=(n_frames * 2.5, 5))
    fig.suptitle('Pixel-Level Anomaly Heatmaps', fontsize=14, fontweight='bold')
    
    for i in range(n_frames):
        # Top row: Original frames
        axes[0, i].imshow(orig_frames[i])
        axes[0, i].set_title(f'Frame {frame_nums[i]}', fontsize=10)
        axes[0, i].axis('off')
        
        # Bottom row: Pure heatmaps (like example image)
        axes[1, i].imshow(heatmaps[i], cmap='hot', vmin=0, vmax=1)
        axes[1, i].set_title(f'{scores_norm[i]:.3f}', fontsize=10)
        axes[1, i].axis('off')
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.show()
    print(f"Saved: {output_path}")
    
    # Also save overlay version
    fig2, axes2 = plt.subplots(2, n_frames, figsize=(n_frames * 2.5, 5))
    fig2.suptitle('Pixel-Level Anomaly Heatmaps (Overlay)', fontsize=14, fontweight='bold')
    
    for i in range(n_frames):
        # Top row: Original frames
        axes2[0, i].imshow(orig_frames[i])
        axes2[0, i].set_title(f'Frame {frame_nums[i]}', fontsize=10)
        axes2[0, i].axis('off')
        
        # Bottom row: Heatmap overlaid on original
        heatmap_colored = cv2.applyColorMap(
            (heatmaps[i] * 255).astype(np.uint8), 
            cv2.COLORMAP_JET
        )
        heatmap_colored = cv2.cvtColor(heatmap_colored, cv2.COLOR_BGR2RGB)
        
        # Blend with original
        alpha = 0.5
        overlay = cv2.addWeighted(orig_frames[i], 1-alpha, heatmap_colored, alpha, 0)
        
        axes2[1, i].imshow(overlay)
        axes2[1, i].set_title(f'{scores_norm[i]:.3f}', fontsize=10)
        axes2[1, i].axis('off')
    
    plt.tight_layout()
    overlay_path = output_path.replace('.png', '_overlay.png')
    plt.savefig(overlay_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saved: {overlay_path}")
    
    return heatmaps, scores_norm


# ===============================
# t-SNE VISUALIZATION
# ===============================

def visualize_tsne(features, scores, threshold, output_path="tsne_visualization.png",
                   perplexity=30, max_samples=2000):
    """
    Generate t-SNE 2D visualization of features colored by anomaly status.
    
    Args:
        features: (N, D) feature tensor
        scores: (N,) normalized anomaly scores  
        threshold: Anomaly threshold value
        output_path: Output image path
        perplexity: t-SNE perplexity parameter
        max_samples: Maximum samples to visualize (for speed)
    """
    from sklearn.manifold import TSNE
    
    print(f"\n{'='*50}")
    print(f"Generating t-SNE Visualization")
    print(f"{'='*50}")
    
    # Convert to numpy if tensor
    if isinstance(features, torch.Tensor):
        features_np = features.numpy()
    else:
        features_np = features
    
    if isinstance(scores, torch.Tensor):
        scores_np = scores.numpy()
    else:
        scores_np = np.array(scores)
    
    n_samples = len(features_np)
    print(f"Total samples: {n_samples}")
    
    # Subsample if too many points (t-SNE is slow for large datasets)
    if n_samples > max_samples:
        print(f"Subsampling to {max_samples} samples for t-SNE...")
        indices = np.random.choice(n_samples, max_samples, replace=False)
        indices = np.sort(indices)
        features_np = features_np[indices]
        scores_np = scores_np[indices]
        n_samples = max_samples
    
    # Classify as normal/anomalous
    is_anomaly = scores_np > threshold
    n_anomalies = is_anomaly.sum()
    n_normal = n_samples - n_anomalies
    
    print(f"Normal: {n_normal}, Anomalous: {n_anomalies}")
    
    # Run t-SNE
    print(f"Running t-SNE (perplexity={perplexity})...")
    tsne = TSNE(
        n_components=2,
        perplexity=min(perplexity, n_samples - 1),
        random_state=SEED,
        n_iter=1000,
        learning_rate='auto',
        init='pca'
    )
    
    embeddings = tsne.fit_transform(features_np)
    print("t-SNE complete!")
    
    # Create visualization
    fig, axes = plt.subplots(1, 2, figsize=(16, 7))
    
    # Plot 1: Binary coloring (Normal vs Anomaly)
    ax1 = axes[0]
    
    # Plot normal points first (so anomalies are on top)
    normal_mask = ~is_anomaly
    ax1.scatter(embeddings[normal_mask, 0], embeddings[normal_mask, 1],
                c='#2E86AB', s=15, alpha=0.6, label=f'Normal ({n_normal})')
    ax1.scatter(embeddings[is_anomaly, 0], embeddings[is_anomaly, 1],
                c='#E63946', s=25, alpha=0.8, label=f'Anomaly ({n_anomalies})')
    
    ax1.set_xlabel('t-SNE Dimension 1', fontsize=12)
    ax1.set_ylabel('t-SNE Dimension 2', fontsize=12)
    ax1.set_title('t-SNE: Normal vs Anomalous Frames', fontsize=14, fontweight='bold')
    ax1.legend(loc='best', fontsize=11)
    ax1.grid(True, alpha=0.3)
    
    # Plot 2: Continuous coloring by anomaly score
    ax2 = axes[1]
    scatter = ax2.scatter(embeddings[:, 0], embeddings[:, 1],
                          c=scores_np, cmap='RdYlGn_r', s=15, alpha=0.7)
    
    # Add threshold line in colorbar
    cbar = plt.colorbar(scatter, ax=ax2)
    cbar.set_label('Anomaly Score', fontsize=12)
    cbar.ax.axhline(y=threshold, color='black', linewidth=2, linestyle='--')
    
    ax2.set_xlabel('t-SNE Dimension 1', fontsize=12)
    ax2.set_ylabel('t-SNE Dimension 2', fontsize=12)
    ax2.set_title('t-SNE: Colored by Anomaly Score', fontsize=14, fontweight='bold')
    ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.show()
    print(f"Saved: {output_path}")
    
    # Print some statistics
    print(f"\nEmbedding Statistics:")
    print(f"  Normal centroid:  ({embeddings[normal_mask, 0].mean():.2f}, {embeddings[normal_mask, 1].mean():.2f})")
    if n_anomalies > 0:
        print(f"  Anomaly centroid: ({embeddings[is_anomaly, 0].mean():.2f}, {embeddings[is_anomaly, 1].mean():.2f})")
    
    return embeddings, is_anomaly


# ===============================
# TRAINING (Vectorized)
# ===============================

def train():
    print("\n" + "=" * 60)
    print("  TRAINING (Vectorized)")
    print("=" * 60)
    
    # Extract features
    features, clips, videos, stats = extract_features_vectorized(
        TRAIN_DIR, "train", stride=2
    )
    
    n_samples = len(features)
    print(f"\nTraining samples: {n_samples}")
    
    # VECTORIZED: Use TensorDataset for efficient batching
    train_dataset = TensorDataset(features)
    train_loader = DataLoader(
        train_dataset,
        batch_size=TRAIN_BATCH,
        shuffle=True,
        num_workers=0,  # Data already in memory
        pin_memory=True,
        drop_last=True
    )
    
    # Model
    model = SimpleAutoencoder(input_dim=2048, hidden_dim=1024, latent_dim=128).to(device)
    
    # Compile model for speed (PyTorch 2.0+)
    if hasattr(torch, 'compile'):
        try:
            model = torch.compile(model, mode='reduce-overhead')
            print("Model compiled with torch.compile()")
        except Exception:
            pass
    
    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Parameters: {n_params:,}")
    
    # Optimizer
    optimizer = optim.AdamW(model.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=NUM_EPOCHS, eta_min=1e-6)
    
    print(f"\nTraining for {NUM_EPOCHS} epochs...")
    
    best_loss = float('inf')
    losses = []
    
    for epoch in range(NUM_EPOCHS):
        model.train()
        epoch_loss = 0.0
        n_batches = 0
        
        for (batch,) in train_loader:
            batch = batch.to(device, non_blocking=True)
            
            optimizer.zero_grad(set_to_none=True)
            
            # VECTORIZED: Forward + loss in one call
            _, loss = model.compute_loss(batch)
            
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            
            epoch_loss += loss.item()
            n_batches += 1
        
        scheduler.step()
        
        avg_loss = epoch_loss / n_batches
        losses.append(avg_loss)
        
        if (epoch + 1) % 10 == 0 or epoch == 0:
            print(f"Epoch {epoch+1:3d}/{NUM_EPOCHS} | Loss: {avg_loss:.6f} | LR: {scheduler.get_last_lr()[0]:.2e}")
        
        if avg_loss < best_loss:
            best_loss = avg_loss
            torch.save(model.state_dict(), os.path.join(FEATURE_CACHE, 'best_model.pth'))
    
    # Load best
    model.load_state_dict(torch.load(os.path.join(FEATURE_CACHE, 'best_model.pth'), weights_only=True))
    print(f"\nBest loss: {best_loss:.6f}")
    
    # Plot
    plt.figure(figsize=(10, 4))
    plt.plot(losses, 'b-', linewidth=2)
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Training Loss')
    plt.grid(True, alpha=0.3)
    plt.savefig('training_loss.png', dpi=150)
    plt.close()
    
    return model, stats


# ===============================
# EVALUATION (Vectorized)
# ===============================

def evaluate(model, train_stats):
    print("\n" + "=" * 60)
    print("  EVALUATION (Vectorized)")
    print("=" * 60)
    
    # Extract test features
    features, clips, videos = extract_test_features_vectorized(
        TEST_DIR, train_stats, stride=1
    )
    
    model.eval()
    
    n_clips = len(features)
    print(f"\nTest clips: {n_clips}")
    
    # VECTORIZED: Compute all errors in batches
    all_errors = torch.zeros(n_clips, dtype=torch.float32)
    
    print("Computing reconstruction errors...")
    
    with torch.no_grad():
        for i in tqdm(range(0, n_clips, TRAIN_BATCH * 2), desc="Evaluating"):
            batch = features[i:i + TRAIN_BATCH * 2].to(device)
            
            # VECTORIZED: Get per-sample errors
            errors = model.compute_errors_vectorized(batch)
            all_errors[i:i + len(errors)] = errors.cpu()
    
    errors_np = all_errors.numpy()
    
    # VECTORIZED: Build frame score mapping using numpy operations
    print("Aggregating frame scores...")
    
    # Pre-allocate score arrays per video
    video_scores = {}
    video_frame_ids = {}
    
    # First, build frame -> error mapping efficiently
    frame_error_map = defaultdict(list)
    
    for i, clip in enumerate(clips):
        vid_id = clip['vid']
        frame_nums = clip['nums']
        error = errors_np[i]
        
        # Assign to center frames (more accurate)
        mid_start = SEQ_LEN // 4
        mid_end = SEQ_LEN - SEQ_LEN // 4
        
        for j in range(mid_start, mid_end):
            frame_error_map[(vid_id, frame_nums[j])].append(error)
    
    # VECTORIZED: Aggregate scores per video
    all_frame_ids = []
    all_scores = []
    
    for vid_id, info in sorted(videos.items()):
        n_frames = info['n']
        frame_nums = info['nums']
        
        # Pre-allocate score array
        scores = np.zeros(n_frames, dtype=np.float32)
        
        for i, fn in enumerate(frame_nums):
            key = (vid_id, fn)
            if key in frame_error_map:
                scores[i] = np.max(frame_error_map[key])  # MAX for anomalies
        
        # VECTORIZED: Temporal smoothing using scipy
        from scipy.ndimage import gaussian_filter1d
        scores = gaussian_filter1d(scores.astype(np.float64), sigma=3)
        
        video_scores[vid_id] = scores
        
        # Build frame IDs
        frame_ids = [f"{vid_id}_{fn}" for fn in frame_nums]
        video_frame_ids[vid_id] = frame_ids
        
        all_frame_ids.extend(frame_ids)
        all_scores.extend(scores)
    
    # VECTORIZED: Convert to numpy array
    all_scores = np.array(all_scores, dtype=np.float64)
    
    # VECTORIZED: Normalize using percentiles
    p1, p99 = np.percentile(all_scores, [1, 99])
    all_scores_norm = np.clip(all_scores, p1, p99)
    all_scores_norm = (all_scores_norm - p1) / (p99 - p1 + 1e-8)
    
    # Compute threshold
    threshold = all_scores_norm.mean() + 2 * all_scores_norm.std()
    n_anomalies = (all_scores_norm > threshold).sum()
    
    print(f"\nResults:")
    print(f"  Frames: {len(all_scores_norm)}")
    print(f"  Score range: [{all_scores_norm.min():.4f}, {all_scores_norm.max():.4f}]")
    print(f"  Threshold: {threshold:.4f}")
    print(f"  Anomalies: {n_anomalies} ({100*n_anomalies/len(all_scores_norm):.1f}%)")
    
    # Save results
    df = pd.DataFrame({'Id': all_frame_ids, 'Score': all_scores_norm})
    df.to_csv(OUTPUT_CSV, index=False)
    print(f"\nSaved: {OUTPUT_CSV}")
    
    # Try to compute AUC
    try:
        from sklearn.metrics import roc_auc_score
        from scipy.io import loadmat
        
        gt_paths = [
            "/kaggle/input/avenue-dataset/Avenue_Corrupted/testing_label",
            "/kaggle/input/avenue-dataset/testing_label",
            os.path.join(DATA_ROOT, "..", "testing_label"),
        ]
        
        gt_dir = None
        for path in gt_paths:
            if os.path.exists(path):
                gt_dir = path
                break
        
        if gt_dir:
            print(f"\nComputing AUC (GT: {gt_dir})")
            
            aucs = []
            for vid_id, scores in video_scores.items():
                mat_files = glob.glob(os.path.join(gt_dir, f"*{vid_id}*.mat"))
                if not mat_files:
                    mat_files = glob.glob(os.path.join(gt_dir, f"*{vid_id:02d}*.mat"))
                
                if mat_files:
                    mat = loadmat(mat_files[0])
                    for key in mat:
                        if not key.startswith('__'):
                            gt = np.array(mat[key]).flatten()
                            break
                    
                    # Normalize
                    s = scores.copy()
                    if s.max() > s.min():
                        s = (s - s.min()) / (s.max() - s.min())
                    
                    min_len = min(len(s), len(gt))
                    s, gt = s[:min_len], gt[:min_len]
                    
                    if len(np.unique(gt)) > 1:
                        auc = roc_auc_score(gt, s)
                        aucs.append(auc)
                        print(f"  Video {vid_id}: AUC = {auc:.4f}")
            
            if aucs:
                print(f"\n{'='*40}")
                print(f"  MEAN AUC: {np.mean(aucs):.4f}")
                print(f"{'='*40}")
    except Exception as e:
        print(f"  Could not compute AUC: {e}")
    
    # Visualization
    fig, axes = plt.subplots(2, 1, figsize=(16, 8))
    
    axes[0].plot(all_scores_norm, color='#E63946', linewidth=0.8)
    axes[0].axhline(threshold, color='#F4A261', linestyle='--', linewidth=2)
    axes[0].fill_between(range(len(all_scores_norm)), all_scores_norm, threshold,
                          where=all_scores_norm > threshold, alpha=0.4, color='#E63946')
    axes[0].set_title("Anomaly Scores (I3D + Autoencoder)", fontsize=14)
    axes[0].set_xlabel("Frame")
    axes[0].set_ylabel("Score")
    axes[0].grid(True, alpha=0.3)
    
    axes[1].hist(all_scores_norm, bins=100, color='#457B9D', alpha=0.7)
    axes[1].axvline(threshold, color='#E63946', linestyle='--', linewidth=2)
    axes[1].set_title("Score Distribution")
    axes[1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('anomaly_analysis.png', dpi=150)
    plt.show()
    
    # Generate pixel-level heatmaps if enabled
    if ENABLE_HEATMAPS:
        visualize_anomaly_heatmaps(
            model, train_stats, TEST_DIR,
            output_path='pixel_heatmaps.png',
            n_frames=HEATMAP_FRAMES,
            video_idx=HEATMAP_VIDEO_IDX
        )
    
    # Generate t-SNE visualization if enabled
    if ENABLE_TSNE:
        visualize_tsne(
            features, all_scores_norm, threshold,
            output_path='tsne_visualization.png',
            perplexity=TSNE_PERPLEXITY,
            max_samples=TSNE_MAX_SAMPLES
        )
    
    return all_scores_norm, threshold


# ===============================
# MAIN
# ===============================

if __name__ == "__main__":
    print("\n" + "=" * 60)
    print("  AVENUE ANOMALY DETECTION - VECTORIZED")
    print("  I3D + Simple Autoencoder + Proper Normalization")
    print("=" * 60)
    
    start = time.time()
    
    # Check dependencies
    try:
        import pytorchvideo
        from sklearn.metrics import roc_auc_score
        print("\n✓ Dependencies OK")
    except ImportError:
        os.system("pip install pytorchvideo scikit-learn -q")
    
    # Train
    model, train_stats = train()
    
    # Evaluate
    scores, threshold = evaluate(model, train_stats)
    
    # Summary
    print(f"\n{'='*60}")
    print(f"  Done! Time: {(time.time() - start)/60:.1f} min")
    print("=" * 60)
    print("\nVectorization applied:")
    print("  ✓ Parallel image loading (ThreadPoolExecutor)")
    print("  ✓ Batch normalization with broadcasting")
    print("  ✓ Pre-allocated tensors for features")
    print("  ✓ TensorDataset for efficient batching")
    print("  ✓ Vectorized MSE computation")
    print("  ✓ torch.compile() model optimization")
    print("  ✓ Vectorized score aggregation")
