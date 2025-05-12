import numpy as np
import tensorflow as tf
from tensorflow import keras
from sklearn.model_selection import train_test_split
import wandb
import os
from datetime import datetime
import matplotlib.pyplot as plt
import io

class WandbCallback(keras.callbacks.Callback):
    # def __init__(self):
    #     super().__init__()
    #     self.losses = []
    #     self.val_losses = []
    #     self.maes = []
    #     self.val_maes = []

    def on_epoch_end(self, epoch, logs=None):
        # Store metrics
        # self.losses.append(logs['loss'])
        # self.val_losses.append(logs['val_loss'])
        # self.maes.append(logs['mae'])
        # self.val_maes.append(logs['val_mae'])
        
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

def log_learningcurves(history , wandb): 
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))
    
    # Plot loss
    ax1.plot(history.history['loss'], label='Training Loss')
    ax1.plot(history.history['val_loss'], label='Validation Loss')
    ax1.set_title('Model Loss')
    ax1.set_xlabel('Epoch')
    ax1.set_ylabel('Loss')
    ax1.legend()
    
    # Plot MAE
    ax2.plot(history.history['mae'], label='Training MAE')
    ax2.plot(history.history['val_mae'], label='Validation MAE')
    ax2.set_title('Model MAE')
    ax2.set_xlabel('Epoch')
    ax2.set_ylabel('MAE')
    ax2.legend()
    
    plt.tight_layout()
    
    # Convert plot to image
    buf = io.BytesIO()
    plt.savefig(buf, format='png', bbox_inches='tight')
    buf.seek(0)
    
    # Convert to numpy array
    import PIL.Image
    image = PIL.Image.open(buf)
    image_array = np.array(image)
    
    # Log final learning curves to wandb
    wandb.log({
        "final_learning_curves": wandb.Image(image_array)
    })
    plt.close()