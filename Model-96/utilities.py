import wandb
import keras
import numpy as np
import matplotlib.pyplot as plt
import io

class WandbCallback(keras.callbacks.Callback):
    def __init__(self):
        super().__init__()
        self.losses = []
        self.val_losses = []
        self.maes = []
        self.val_maes = []

    def on_epoch_end(self, epoch, logs=None):
        # Store metrics
        self.losses.append(logs['loss'])
        self.val_losses.append(logs['val_loss'])
        self.maes.append(logs['mae'])
        self.val_maes.append(logs['val_mae'])
        
        # Log metrics to wandb
        wandb.log({
            "epoch": epoch,
            "train_loss": logs['loss'],
            "val_loss": logs['val_loss'],
            "train_mae": logs['mae'],
            "val_mae": logs['val_mae']
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

def load_dataset_with_weights(npz_path):
    """
    Load a .npz containing 'features' and 'poses' (in YPR order),
    compute per-sample weights based on head-off-axis angle,
    and return an extended dict with a 'weights' field.

    Weighting follows Eq. (12–13):
      δ = arccos( cos(pitch) * cos(yaw) )             (12)  [oai_citation:0‡TMM_202201_HeadPose.pdf](file-service://file-1tYGoLsc7AfuTBj2j9aPZR)
      w = 1                    if δ ≤ 60°
          0.5 ** ((δ - 60) / 5)  if δ  > 60°          (13)  [oai_citation:1‡TMM_202201_HeadPose.pdf](file-service://file-1tYGoLsc7AfuTBj2j9aPZR)
    """
    data = np.load(npz_path)
    features = data['features']
    poses    = data['poses']   # shape (N,3): [yaw, pitch, roll]

    # Extract yaw & pitch (in degrees), convert to radians
    yaw_rad   = np.deg2rad(poses[:, 0])
    pitch_rad = np.deg2rad(poses[:, 1])

    # Eq. 12: δ = arccos( cos(pitch) * cos(yaw) )
    cos_prod  = np.cos(pitch_rad) * np.cos(yaw_rad)
    cos_prod  = np.clip(cos_prod, -1.0, 1.0)  # numerical safety
    delta_rad = np.arccos(cos_prod)
    delta_deg = np.rad2deg(delta_rad)

    # Eq. 13: weight = 1 if δ ≤ 60; else weight = 0.5 ** ((δ − 60) / 5)
    weights = np.ones_like(delta_deg)
    mask    = delta_deg > 60.0
    weights[mask] = 0.5 ** ((delta_deg[mask] - 60.0) / 5.0)

    return {
        'features': features,
        'poses':    poses,
        'weights':  weights
    }
    

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