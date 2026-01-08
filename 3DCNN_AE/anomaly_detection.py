# ============================================================================
# Video Anomaly Detection - Multi-Layer Reconstruction Autoencoder
# Based on: "Video anomaly detection based on a multi-layer reconstruction 
# autoencoder with a variance attention strategy" (2024)
# ============================================================================

import os
import glob
import numpy as np
import pandas as pd
from PIL import Image
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import pytorch_lightning as pl
from pytorch_lightning.callbacks import EarlyStopping, ModelCheckpoint
import warnings
warnings.filterwarnings('ignore')

# ============================================================================
# CONFIGURATION
# ============================================================================

class Config:
    # Kaggle dataset paths - Updated for your specific dataset structure
    BASE_PATH = '/kaggle/input/pixel-play-26/Avenue_Corrupted-20251221T112159Z-3-001/Avenue_Corrupted/Dataset'
    TRAIN_PATH = f'{BASE_PATH}/training_videos'  # Training .jpg frames
    TEST_PATH = f'{BASE_PATH}/testing_videos'    # Testing .jpg frames
    
    # Model parameters
    IMG_HEIGHT = 368  # Padded from 360 to be divisible by 16
    IMG_WIDTH = 640   # Divisible by 16
    TOKEN_HEIGHT = 48
    TOKEN_WIDTH = 48
    SEQUENCE_LENGTH = 8
    CHANNELS = 3
    
    # Training parameters
    BATCH_SIZE = 16
    MAX_EPOCHS = 80      
    LEARNING_RATE = 0.0001
    WEIGHT_DECAY = 0.0001
    PATIENCE = 8         
    
    # Loss weights
    ALPHA_1 = 1.0   # Appearance loss weight
    ALPHA_2 = 25.0  # Motion loss weight
    
    # Score weights (Avenue dataset specific)
    BETA_1 = -0.1   # Appearance score weight
    BETA_2 = 2.0    # Motion score weight
    
    # Multi-layer reconstruction layers (exclude bottleneck layer 4)
    RECON_LAYERS = [0, 2, 3]  # Input(3ch), enc2(128ch), enc3(256ch)
    
    # Output
    OUTPUT_CSV = 'anomaly_scores.csv'
    CHECKPOINT_DIR = 'checkpoints'

config = Config()

# ============================================================================
# EARLY DATASET VALIDATION
# ============================================================================

def get_dataset_frame_counts(train_path, test_path):
    """
    Retrieve the number of frames in train and test datasets
    This function runs early in the pipeline to validate data availability
    
    Args:
        train_path: Path to training dataset
        test_path: Path to testing dataset
    
    Returns:
        dict: Dictionary containing frame counts for train and test sets
    """
    print("\n" + "="*60)
    print("EARLY DATASET VALIDATION - Counting Frames")
    print("="*60)
    
    results = {
        'train': {'videos': 0, 'total_frames': 0, 'frames_per_video': []},
        'test': {'videos': 0, 'total_frames': 0, 'frames_per_video': []}
    }
    
    # Count training frames
    if os.path.exists(train_path):
        train_videos = sorted([d for d in os.listdir(train_path) 
                              if os.path.isdir(os.path.join(train_path, d))])
        results['train']['videos'] = len(train_videos)
        
        for video_folder in train_videos:
            video_path = os.path.join(train_path, video_folder)
            frames = glob.glob(os.path.join(video_path, '*.jpg'))
            frame_count = len(frames)
            results['train']['frames_per_video'].append(frame_count)
            results['train']['total_frames'] += frame_count
        
        print(f"\n📊 TRAINING DATASET:")
        print(f"   - Number of videos: {results['train']['videos']}")
        print(f"   - Total frames: {results['train']['total_frames']}")
        print(f"   - Average frames per video: {np.mean(results['train']['frames_per_video']):.1f}")
        print(f"   - Min frames: {min(results['train']['frames_per_video']) if results['train']['frames_per_video'] else 0}")
        print(f"   - Max frames: {max(results['train']['frames_per_video']) if results['train']['frames_per_video'] else 0}")
    else:
        print(f"\n⚠️  WARNING: Training path not found: {train_path}")
    
    # Count testing frames
    if os.path.exists(test_path):
        test_videos = sorted([d for d in os.listdir(test_path) 
                             if os.path.isdir(os.path.join(test_path, d))])
        results['test']['videos'] = len(test_videos)
        
        for video_folder in test_videos:
            video_path = os.path.join(test_path, video_folder)
            frames = glob.glob(os.path.join(video_path, '*.jpg'))
            frame_count = len(frames)
            results['test']['frames_per_video'].append(frame_count)
            results['test']['total_frames'] += frame_count
        
        print(f"\n📊 TESTING DATASET:")
        print(f"   - Number of videos: {results['test']['videos']}")
        print(f"   - Total frames: {results['test']['total_frames']}")
        print(f"   - Average frames per video: {np.mean(results['test']['frames_per_video']):.1f}")
        print(f"   - Min frames: {min(results['test']['frames_per_video']) if results['test']['frames_per_video'] else 0}")
        print(f"   - Max frames: {max(results['test']['frames_per_video']) if results['test']['frames_per_video'] else 0}")
    else:
        print(f"\n⚠️  WARNING: Testing path not found: {test_path}")
    
    print("\n" + "="*60)
    print("✓ Frame count validation completed")
    print("="*60 + "\n")
    
    return results

# ============================================================================
# DATASET
# ============================================================================

class AvenueDataset(Dataset):
    """Dataset for Avenue video frames (.jpg files)"""
    
    def __init__(self, root_dir, sequence_length=8, mode='train', transform=None):
        self.root_dir = root_dir
        self.sequence_length = sequence_length
        self.mode = mode
        self.transform = transform
        
        # Get all video folders
        self.video_folders = sorted([d for d in os.listdir(root_dir) 
                                     if os.path.isdir(os.path.join(root_dir, d))])
        
        # Build frame sequences
        self.sequences = []
        self.video_ids = []
        
        for video_id, video_folder in enumerate(self.video_folders):
            video_path = os.path.join(root_dir, video_folder)
            frames = sorted(glob.glob(os.path.join(video_path, '*.jpg')))
            
            # Create sequences
            for i in range(0, len(frames) - sequence_length + 1, sequence_length if mode == 'train' else 1):
                seq_frames = frames[i:i + sequence_length]
                if len(seq_frames) == sequence_length:
                    self.sequences.append(seq_frames)
                    self.video_ids.append(video_id)
        
        print(f"{mode.upper()} dataset: {len(self.sequences)} sequences from {len(self.video_folders)} videos")
    
    def __len__(self):
        return len(self.sequences)
    
    def __getitem__(self, idx):
        frame_paths = self.sequences[idx]
        video_id = self.video_ids[idx]
        
        # Load frames
        frames = []
        for frame_path in frame_paths:
            img = Image.open(frame_path).convert('RGB')
            img = img.resize((config.IMG_WIDTH, config.IMG_HEIGHT))
            img = np.array(img) / 255.0  # Normalize to [0, 1]
            frames.append(img)
        
        # Stack frames: (T, H, W, C)
        frames = np.stack(frames, axis=0)
        
        # Convert to torch tensor: (C, T, H, W)
        frames = torch.from_numpy(frames).float().permute(3, 0, 1, 2)
        
        return {
            'frames': frames,
            'video_id': video_id,
            'frame_idx': idx
        }

# ============================================================================
# NETWORK ARCHITECTURE
# ============================================================================

class DownsamplingBlock(nn.Module):
    """3D Downsampling block with convolutions"""
    
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.conv1 = nn.Conv3d(in_channels, out_channels, kernel_size=3, stride=1, padding=1)
        self.bn1 = nn.BatchNorm3d(out_channels)
        self.conv2 = nn.Conv3d(out_channels, out_channels, kernel_size=3, stride=1, padding=1)
        self.bn2 = nn.BatchNorm3d(out_channels)
        self.conv3 = nn.Conv3d(out_channels, out_channels, kernel_size=1, stride=2, padding=0)
        self.bn3 = nn.BatchNorm3d(out_channels)
        self.relu = nn.ReLU(inplace=True)
    
    def forward(self, x):
        x = self.relu(self.bn1(self.conv1(x)))
        x = self.relu(self.bn2(self.conv2(x)))
        x = self.relu(self.bn3(self.conv3(x)))
        return x

class UpsamplingBlock(nn.Module):
    """3D Upsampling block with convolutions"""
    
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.conv1 = nn.Conv3d(in_channels, out_channels, kernel_size=5, stride=1, padding=2)
        self.bn1 = nn.BatchNorm3d(out_channels)
        self.conv2 = nn.Conv3d(out_channels, out_channels, kernel_size=5, stride=1, padding=2)
        self.bn2 = nn.BatchNorm3d(out_channels)
        self.upsample = nn.Upsample(scale_factor=2, mode='trilinear', align_corners=True)
        self.conv3 = nn.Conv3d(out_channels, out_channels, kernel_size=3, stride=1, padding=1)
        self.bn3 = nn.BatchNorm3d(out_channels)
        self.relu = nn.ReLU(inplace=True)
    
    def forward(self, x):
        x = self.relu(self.bn1(self.conv1(x)))
        x = self.relu(self.bn2(self.conv2(x)))
        x = self.upsample(x)
        x = self.relu(self.bn3(self.conv3(x)))
        return x

class MultiLayerAutoencoder(nn.Module):
    """Multi-layer reconstruction autoencoder"""
    
    def __init__(self, in_channels=3):
        super().__init__()
        
        # Encoder layers
        self.enc1 = DownsamplingBlock(in_channels, 64)
        self.enc2 = DownsamplingBlock(64, 128)
        self.enc3 = DownsamplingBlock(128, 256)
        self.enc4 = DownsamplingBlock(256, 512)
        
        # Decoder layers
        self.dec4 = UpsamplingBlock(512, 256)
        self.dec3 = UpsamplingBlock(256, 128)
        self.dec2 = UpsamplingBlock(128, 64)
        self.dec1 = UpsamplingBlock(64, in_channels)
        
        # Final output layer
        self.final = nn.Conv3d(in_channels, in_channels, kernel_size=1)
    
    def forward(self, x):
        # Store intermediate features
        features = {'enc': [x], 'dec': []}
        
        # Encoder
        x1 = self.enc1(x)
        features['enc'].append(x1)
        
        x2 = self.enc2(x1)
        features['enc'].append(x2)
        
        x3 = self.enc3(x2)
        features['enc'].append(x3)
        
        x4 = self.enc4(x3)
        features['enc'].append(x4)
        
        # Decoder
        x = self.dec4(x4)
        features['dec'].append(x)
        
        x = self.dec3(x)
        features['dec'].append(x)
        
        x = self.dec2(x)
        features['dec'].append(x)
        
        x = self.dec1(x)
        features['dec'].append(x)
        
        x = self.final(x)
        features['dec'].insert(0, x)
        
        return x, features

# ============================================================================
# VARIANCE ATTENTION
# ============================================================================

def compute_variance_attention(feature_map):
    """
    Compute variance attention along channel and temporal dimensions
    Args:
        feature_map: (B, C, T, H, W)
    Returns:
        channel_attention: (B, 1, T, H, W)
        temporal_attention: (B, C, 1, H, W)
    """
    B, C, T, H, W = feature_map.shape
    
    # Channel variance attention (along channel dimension)
    channel_mean = feature_map.mean(dim=1, keepdim=True)  # (B, 1, T, H, W)
    channel_variance = ((feature_map - channel_mean) ** 2).mean(dim=1, keepdim=True)  # (B, 1, T, H, W)
    
    # Reshape for softmax over spatial locations
    channel_variance_flat = channel_variance.view(B, T, H * W)
    channel_attention_flat = F.softmax(channel_variance_flat, dim=-1)
    channel_attention = channel_attention_flat.view(B, 1, T, H, W)
    
    # Temporal variance attention (along temporal dimension)
    temporal_mean = feature_map.mean(dim=2, keepdim=True)  # (B, C, 1, H, W)
    temporal_variance = ((feature_map - temporal_mean) ** 2).mean(dim=2, keepdim=True)  # (B, C, 1, H, W)
    
    # Reshape for softmax over spatial locations
    temporal_variance_flat = temporal_variance.view(B, C, H * W)
    temporal_attention_flat = F.softmax(temporal_variance_flat, dim=-1)
    temporal_attention = temporal_attention_flat.view(B, C, 1, H, W)
    
    return channel_attention, temporal_attention

# ============================================================================
# LOSS FUNCTIONS
# ============================================================================

def appearance_loss(reconstruction, target, attention_weights=None):
    """Compute appearance loss with optional attention weighting"""
    loss = F.mse_loss(reconstruction, target, reduction='none')
    
    if attention_weights is not None:
        loss = loss * attention_weights
    
    return loss.mean()

def motion_loss(reconstruction, target, attention_weights=None):
    """
    Compute motion loss using temporal gradients
    Args:
        reconstruction: (B, C, T, H, W)
        target: (B, C, T, H, W)
    """
    # Compute temporal gradients
    target_gradient = torch.abs(target[:, :, 1:] - target[:, :, :-1])
    recon_gradient = torch.abs(reconstruction[:, :, 1:] - reconstruction[:, :, :-1])
    
    loss = F.mse_loss(recon_gradient, target_gradient, reduction='none')
    
    if attention_weights is not None:
        # Adjust attention for temporal dimension
        attention_weights = attention_weights[:, :, 1:]
        loss = loss * attention_weights
    
    return loss.mean()

# ============================================================================
# PYTORCH LIGHTNING MODULE
# ============================================================================

class AnomalyDetectionModel(pl.LightningModule):
    """PyTorch Lightning module for anomaly detection"""
    
    def __init__(self, config):
        super().__init__()
        self.save_hyperparameters()
        self.config = config
        
        # Build model
        self.model = MultiLayerAutoencoder(in_channels=config.CHANNELS)
        
        # For validation metrics
        self.validation_step_outputs = []
    
    def forward(self, x):
        return self.model(x)
    
    def compute_multi_layer_loss(self, features, target, mode='train'):
        """Compute multi-layer reconstruction loss"""
        total_app_loss = 0.0
        total_mot_loss = 0.0
        
        # Map decoder features to encoder features properly
        # features['enc'] = [input(3ch), enc1(64ch), enc2(128ch), enc3(256ch), enc4(512ch)]
        # features['dec'] = [final(3ch), dec4(256ch), dec3(128ch), dec2(64ch), dec1(3ch)]
        # Proper mapping: enc[0]→dec[0], enc[1]→dec[3], enc[2]→dec[2], enc[3]→dec[1], enc[4]→bottleneck
        
        dec_to_enc_map = {
            0: 0,  # input (3ch) → final output (3ch)
            1: 3,  # enc1 (64ch) → dec2 output (64ch)
            2: 2,  # enc2 (128ch) → dec3 output (128ch) 
            3: 1,  # enc3 (256ch) → dec4 output (256ch)
            4: 1   # enc4 (512ch) → dec4 (256ch) - will be interpolated
        }
        
        for layer_idx in self.config.RECON_LAYERS:
            enc_feat = features['enc'][layer_idx]
            dec_feat = features['dec'][dec_to_enc_map[layer_idx]]
            
            # Resize if necessary
            if enc_feat.shape != dec_feat.shape:
                dec_feat = F.interpolate(dec_feat, size=enc_feat.shape[2:], 
                                        mode='trilinear', align_corners=True)
            
            # Compute variance attention
            channel_att, temporal_att = compute_variance_attention(enc_feat)
            attention = channel_att + temporal_att
            
            # Appearance loss
            app_loss = appearance_loss(dec_feat, enc_feat, attention)
            total_app_loss += app_loss
            
            # Motion loss
            mot_loss = motion_loss(dec_feat, enc_feat, attention)
            total_mot_loss += mot_loss
        
        # Average over layers
        total_app_loss /= len(self.config.RECON_LAYERS)
        total_mot_loss /= len(self.config.RECON_LAYERS)
        
        # Combined loss
        total_loss = self.config.ALPHA_1 * total_app_loss + self.config.ALPHA_2 * total_mot_loss
        
        return total_loss, total_app_loss, total_mot_loss
    
    def training_step(self, batch, batch_idx):
        frames = batch['frames']
        
        # Forward pass
        reconstruction, features = self.model(frames)
        
        # Compute loss
        loss, app_loss, mot_loss = self.compute_multi_layer_loss(features, frames, mode='train')
        
        # Logging
        self.log('train_loss', loss, prog_bar=True)
        self.log('train_app_loss', app_loss)
        self.log('train_mot_loss', mot_loss)
        
        return loss
    
    def validation_step(self, batch, batch_idx):
        frames = batch['frames']
        
        # Forward pass
        reconstruction, features = self.model(frames)
        
        # Compute loss
        loss, app_loss, mot_loss = self.compute_multi_layer_loss(features, frames, mode='val')
        
        # Compute anomaly scores
        dec_to_enc_map = {0: 0, 1: 3, 2: 2, 3: 1, 4: 1}
        
        app_scores = []
        mot_scores = []
        
        for layer_idx in self.config.RECON_LAYERS:
            enc_feat = features['enc'][layer_idx]
            dec_feat = features['dec'][dec_to_enc_map[layer_idx]]
            
            if enc_feat.shape != dec_feat.shape:
                dec_feat = F.interpolate(dec_feat, size=enc_feat.shape[2:], 
                                        mode='trilinear', align_corners=True)
            
            # Frame-level scores
            app_score = F.mse_loss(dec_feat, enc_feat, reduction='none').mean(dim=[1, 3, 4])  # (B, T)
            app_scores.append(app_score)
            
            # Motion scores
            enc_grad = torch.abs(enc_feat[:, :, 1:] - enc_feat[:, :, :-1])
            dec_grad = torch.abs(dec_feat[:, :, 1:] - dec_feat[:, :, :-1])
            mot_score = F.mse_loss(dec_grad, enc_grad, reduction='none').mean(dim=[1, 3, 4])  # (B, T-1)
            mot_scores.append(mot_score)
        
        # Average scores across layers
        app_score = torch.stack(app_scores).mean(dim=0)  # (B, T)
        mot_score = torch.stack(mot_scores).mean(dim=0)  # (B, T-1)
        
        # Store for epoch end
        self.validation_step_outputs.append({
            'loss': loss,
            'app_loss': app_loss,
            'mot_loss': mot_loss,
            'app_score': app_score,
            'mot_score': mot_score
        })
        
        self.log('val_loss', loss, prog_bar=True)
        
        return loss
    
    def on_validation_epoch_end(self):
        avg_loss = torch.stack([x['loss'] for x in self.validation_step_outputs]).mean()
        self.log('val_loss_epoch', avg_loss, prog_bar=True)
        self.validation_step_outputs.clear()
    
    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(
            self.parameters(),
            lr=self.config.LEARNING_RATE,
            weight_decay=self.config.WEIGHT_DECAY
        )
        
        scheduler = torch.optim.lr_scheduler.StepLR(
            optimizer,
            step_size=80,
            gamma=0.5
        )
        
        return {
            'optimizer': optimizer,
            'lr_scheduler': {
                'scheduler': scheduler,
                'interval': 'epoch'
            }
        }

# ============================================================================
# TESTING AND CSV OUTPUT
# ============================================================================

def test_and_generate_csv_by_video(model, test_dataset, config):
    """
    Process each video separately to ensure correct frame numbering
    Output format: Id,Predicted (e.g., 1_1, 1_2, ...)
    """
    model.eval()
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = model.to(device)
    
    # Get unique videos
    video_folders = sorted([d for d in os.listdir(config.TEST_PATH) 
                           if os.path.isdir(os.path.join(config.TEST_PATH, d))])
    
    all_results = []
    
    print("Processing videos...")
    for video_idx, video_folder in enumerate(video_folders, start=1):
        video_path = os.path.join(config.TEST_PATH, video_folder)
        frames_paths = sorted(glob.glob(os.path.join(video_path, '*.jpg')))
        
        print(f"\nVideo {video_idx}/{len(video_folders)}: {video_folder} ({len(frames_paths)} frames)")
        
        video_scores = []
        
        # Process video in sliding windows
        for start_idx in range(0, len(frames_paths) - config.SEQUENCE_LENGTH + 1):
            sequence_paths = frames_paths[start_idx:start_idx + config.SEQUENCE_LENGTH]
            
            # Load frames
            frames_list = []
            for frame_path in sequence_paths:
                img = Image.open(frame_path).convert('RGB')
                img = img.resize((config.IMG_WIDTH, config.IMG_HEIGHT))
                img = np.array(img) / 255.0
                frames_list.append(img)
            
            frames_tensor = np.stack(frames_list, axis=0)
            frames_tensor = torch.from_numpy(frames_tensor).float().permute(3, 0, 1, 2)
            frames_tensor = frames_tensor.unsqueeze(0).to(device)
            
            with torch.no_grad():
                reconstruction, features = model.model(frames_tensor)
                
                dec_to_enc_map = {0: 0, 1: 3, 2: 2, 3: 1, 4: 1}
                app_scores_all = []
                mot_scores_all = []
                
                for layer_idx in config.RECON_LAYERS:
                    enc_feat = features['enc'][layer_idx]
                    dec_feat = features['dec'][dec_to_enc_map[layer_idx]]
                    
                    if enc_feat.shape != dec_feat.shape:
                        dec_feat = F.interpolate(dec_feat, size=enc_feat.shape[2:], 
                                                mode='trilinear', align_corners=True)
                    
                    # Appearance scores
                    app_score = F.mse_loss(dec_feat, enc_feat, reduction='none').mean(dim=[1, 3, 4])
                    app_scores_all.append(app_score)
                    
                    # Motion scores
                    enc_grad = torch.abs(enc_feat[:, :, 1:] - enc_feat[:, :, :-1])
                    dec_grad = torch.abs(dec_feat[:, :, 1:] - dec_feat[:, :, :-1])
                    mot_score = F.mse_loss(dec_grad, enc_grad, reduction='none').mean(dim=[1, 3, 4])
                    mot_scores_all.append(mot_score)
                
                # Average across layers
                app_score = torch.stack(app_scores_all).mean(dim=0).squeeze(0)
                mot_score = torch.stack(mot_scores_all).mean(dim=0).squeeze(0)
                mot_score_padded = F.pad(mot_score, (0, 1), value=mot_score[-1])
                
                for t in range(config.SEQUENCE_LENGTH):
                    frame_score = (config.BETA_1 * app_score[t].item() + 
                                  config.BETA_2 * mot_score_padded[t].item())
                    
                    frame_global_idx = start_idx + t
                    while len(video_scores) <= frame_global_idx:
                        video_scores.append([])
                    video_scores[frame_global_idx].append(frame_score)
        
        for frame_num, scores in enumerate(video_scores, start=1):
            if scores:
                all_results.append({
                    'video_id': video_idx,
                    'frame_num': frame_num,
                    'score': np.mean(scores)
                })
    
    # Convert to DataFrame
    df = pd.DataFrame(all_results)
    
    # Normalize scores to [0, 1]
    min_score = df['score'].min()
    max_score = df['score'].max()
    df['score_normalized'] = (df['score'] - min_score) / (max_score - min_score)
    
    # Create competition format
    df['Id'] = df['video_id'].astype(str) + '_' + df['frame_num'].astype(str)
    df['Predicted'] = df['score_normalized'].round(6)
    
    # Select only required columns and sort
    output_df = df[['Id', 'Predicted']].sort_values('Id')
    
    # Save to CSV
    output_df.to_csv(config.OUTPUT_CSV, index=False)
    print(f"\n{'='*60}")
    print(f"✓ Results saved to {config.OUTPUT_CSV}")
    print(f"✓ Total predictions: {len(output_df)}")
    print(f"✓ Videos: {df['video_id'].nunique()}")
    print(f"{'='*60}")
    
    return output_df

# ============================================================================
# MAIN TRAINING SCRIPT
# ============================================================================

def main():
    # Set random seed
    pl.seed_everything(42)
    
    # EARLY VALIDATION: Count frames in train and test datasets
    frame_counts = get_dataset_frame_counts(config.TRAIN_PATH, config.TEST_PATH)
    
    # Create datasets
    print("Loading datasets...")
    train_dataset = AvenueDataset(
        root_dir=config.TRAIN_PATH,
        sequence_length=config.SEQUENCE_LENGTH,
        mode='train'
    )
    
    # Create data loaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=config.BATCH_SIZE,
        shuffle=True,
        num_workers=2,
        pin_memory=True
    )
    
    # Use 10% of training data for validation
    val_size = int(0.1 * len(train_dataset))
    train_size = len(train_dataset) - val_size
    train_subset, val_subset = torch.utils.data.random_split(
        train_dataset, [train_size, val_size]
    )
    
    val_loader = DataLoader(
        val_subset,
        batch_size=config.BATCH_SIZE,
        shuffle=False,
        num_workers=2,
        pin_memory=True
    )
    
    # Create model
    print("\nBuilding model...")
    model = AnomalyDetectionModel(config)
    
    # Callbacks
    early_stop_callback = EarlyStopping(
        monitor='val_loss',
        patience=config.PATIENCE,
        mode='min',
        verbose=True
    )
    
    checkpoint_callback = ModelCheckpoint(
        dirpath=config.CHECKPOINT_DIR,
        filename='anomaly-{epoch:02d}-{val_loss:.4f}',
        monitor='val_loss',
        mode='min',
        save_top_k=3
    )
    
    # Trainer
    trainer = pl.Trainer(
        max_epochs=config.MAX_EPOCHS,
        callbacks=[early_stop_callback, checkpoint_callback],
        accelerator='auto',
        devices=1,
        precision=16,
        log_every_n_steps=10
    )
    
    # Train
    print("\nStarting training...")
    trainer.fit(model, train_loader, val_loader)
    
    # Test and generate CSV in competition format
    print("\n" + "="*60)
    print("Testing on test dataset...")
    print("="*60)
    
    best_model_path = checkpoint_callback.best_model_path
    print(f"Loading best model from: {best_model_path}")
    
    best_model = AnomalyDetectionModel.load_from_checkpoint(best_model_path, config=config)
    
    test_dataset = AvenueDataset(
        root_dir=config.TEST_PATH,
        sequence_length=config.SEQUENCE_LENGTH,
        mode='test'
    )
    
    # Use video-by-video processing for accurate frame numbering
    results_df = test_and_generate_csv_by_video(best_model, test_dataset, config)
    
    print("\n" + "="*60)
    print("TRAINING AND TESTING COMPLETED!")
    print("="*60)
    
    # Display sample results
    print("\nSample output (first 20 rows):")
    print(results_df.head(20).to_string(index=False))
    
    print("\nSample output (last 20 rows):")
    print(results_df.tail(20).to_string(index=False))
    
    print("\nAnomaly score statistics:")
    print(results_df['Predicted'].describe())

if __name__ == '__main__':
    main()
