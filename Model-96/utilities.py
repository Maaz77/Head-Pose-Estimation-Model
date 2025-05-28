import wandb
import keras
import numpy as np
import matplotlib.pyplot as plt
import io
import os
from typing import List, Tuple

class WandbCallback(keras.callbacks.Callback):

    def on_epoch_end(self, epoch, logs=None):        
        logs = logs or {}
        
        # Calculate mean regression losses
        mean_train_reg_loss = np.mean([
            logs['yaw_reg_loss'],
            logs['pitch_reg_loss'],
            logs['roll_reg_loss']
        ])
        mean_val_reg_loss = np.mean([
            logs['val_yaw_reg_loss'],
            logs['val_pitch_reg_loss'],
            logs['val_roll_reg_loss']
        ])
        # Calculate mean MAE metrics
        mean_train_mae = np.mean([
            logs['yaw_reg_mae'],
            logs['pitch_reg_mae'],
            logs['roll_reg_mae']
        ])
        mean_val_mae = np.mean([
            logs['val_yaw_reg_mae'],
            logs['val_pitch_reg_mae'],
            logs['val_roll_reg_mae']
        ])

        # Log metrics to wandb
        wandb.log({
            "epoch": epoch,
            "total_train_loss": logs['loss'],
            "total_val_loss": logs['val_loss'],
            "avg_train_reg_loss": mean_train_reg_loss,
            "avg_val_reg_loss": mean_val_reg_loss,
            "avg_train_reg_mae": mean_train_mae,
            "avg_val_reg_mae": mean_val_mae
        })

def load_dataset(dataset_path):
    """Load the dataset containing features and poses."""
    data = np.load(dataset_path)
    return data['features'], data['poses']


def load_model_from_json(model_path):
    """Load a model from a JSON file."""
    with open(model_path, 'r') as f:
        model_json = f.read()
    return keras.models.model_from_json(model_json)


def load_and_preprocess(
    npz_path: str,
    n_bins: int = 66,
    M: float = 99.0
):
    """
    Load .npz with 'features' and 'poses' (yaw,pitch,roll in degrees),
    filter out samples with any |angle| > M,
    compute per-sample weights,
    compute discrete bin indices and one-hot labels.
    Returns: X, poses, weights, one_hot (N,3,n_bins)
    """
    data     = np.load(npz_path)
    X        = data['features']             
    poses    = data['poses']               

    # 1) Filter out-of-range samples
    in_range = np.all(np.abs(poses) <= M, axis=1)
    X        = X[in_range]
    poses    = poses[in_range]

    # 2) Compute weights
    yaw_rad   = np.deg2rad(poses[:, 0])
    pitch_rad = np.deg2rad(poses[:, 1])
    cos_prod  = np.cos(pitch_rad) * np.cos(yaw_rad)
    cos_prod  = np.clip(cos_prod, -1.0, 1.0)
    delta_deg = np.rad2deg(np.arccos(cos_prod))

    weights = np.ones_like(delta_deg)
    far_mask = delta_deg > 60.0
    weights[far_mask] = 0.5 ** ((delta_deg[far_mask] - 60.0) / 5.0)

    # 3) Discretize into bins
    delta = 2 * M / n_bins
    eps   = 1e-6
    raw_idx = np.floor((poses + M + eps) / delta).astype(int)
    bin_indices = np.clip(raw_idx, 0, n_bins - 1)  # shape (N,3)

    # 4) One-hot encoding
    one_hot = np.eye(n_bins, dtype=np.float32)[bin_indices]  # (N,3,n_bins)

    return X, poses, weights, one_hot , bin_indices



def analyze_angle_distributions(train_poses, test_poses):
    """Analyze and visualize angle distributions in train and test sets."""
    # Remove duplicates from both sets
    train_unique = np.unique(train_poses, axis=0)
    test_unique = np.unique(test_poses, axis=0)
    
    # Create subplots for each angle
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    fig.suptitle('Angle Distributions in Train and Test Sets', fontsize=16)
    
    angle_names = ['Yaw', 'Pitch', 'Roll']
    
    for idx, (angle_name, ax) in enumerate(zip(angle_names, axes[0])):
        # Plot train set distribution
        ax.hist(train_unique[:, idx], bins=50, alpha=0.5, label='Train', color='blue')
        ax.set_title(f'{angle_name} Distribution')
        ax.set_xlabel('Angle (degrees)')
        ax.set_ylabel('Count')
        ax.legend()
    
    for idx, (angle_name, ax) in enumerate(zip(angle_names, axes[1])):
        # Plot test set distribution
        ax.hist(test_unique[:, idx], bins=50, alpha=0.5, label='Test', color='red')
        ax.set_title(f'{angle_name} Distribution')
        ax.set_xlabel('Angle (degrees)')
        ax.set_ylabel('Count')
        ax.legend()
    
    plt.tight_layout()
    
    # Convert plot to image
    buf = io.BytesIO()
    plt.savefig(buf, format='png', bbox_inches='tight')
    buf.seek(0)
    
    # Convert to numpy array
    import PIL.Image
    image = PIL.Image.open(buf)
    image_array = np.array(image)
    

    wandb.log({
        "angle_distributions": wandb.Image(image_array)
        
    })
    plt.close()
    
def plot_weights_distribution(weights, title):
    plt.figure(figsize=(10, 6))
    plt.scatter(range(len(weights)), weights, alpha=0.7)
    plt.title(title)
    plt.xlabel('Index')
    plt.ylabel('Weights')
    plt.grid()
    plt.show()
    
    
def load_datasets_for_training(dataset_names: List[str] , num_bins: int = 66):
    """
    Load and concatenate multiple datasets for training.
    
    Args:
        dataset_names: List of dataset file names (without .npz extension)
        
    Returns:
        tuple: (train_features, train_poses, train_weights, train_one_hot)
    """
    print(f"Loading {len(dataset_names)} datasets...")
    DIR_PATH = os.getenv('FEATUREMAPS_DIR_PATH')
    
    # Load first dataset to initialize arrays
    if not dataset_names:
        raise ValueError("At least one dataset name must be provided")
    
    first_dataset = dataset_names[0]
    train_features, train_poses, train_weights, train_one_hot, _ = load_and_preprocess(
        DIR_PATH + first_dataset, n_bins=num_bins
    )
    
    # Load and concatenate additional datasets
    for dataset_name in dataset_names[1:]:
        features, poses, weights, one_hot, _ = load_and_preprocess(
            DIR_PATH + dataset_name, n_bins=num_bins
        )
        
        # Concatenate with existing data
        train_features = np.concatenate((train_features, features), axis=0)
        train_poses = np.concatenate((train_poses, poses), axis=0)
        train_weights = np.concatenate((train_weights, weights), axis=0)
        train_one_hot = np.concatenate((train_one_hot, one_hot), axis=0)
        
        # Clean up to free memory
        del features, poses, weights, one_hot
    
    # Reshape features
    train_features = train_features.reshape(-1, 1, 1, 96)
    
    print(f"train_features shape: {train_features.shape}")
    print(f"train_poses shape: {train_poses.shape}")
    print(f"train_weights shape: {train_weights.shape}")
    print(f"train_one_hot shape: {train_one_hot.shape}")
    
    return train_features, train_poses, train_weights, train_one_hot