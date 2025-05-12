import numpy as np
import tensorflow as tf
from tensorflow import keras
from sklearn.model_selection import train_test_split
import wandb
import os
import io
from datetime import datetime
import matplotlib.pyplot as plt
from utilities import WandbCallback, load_dataset, load_dataset_with_weights, load_model_from_json, analyze_angle_distributions

# Suppress warnings
import warnings

warnings.filterwarnings('ignore')
tf.get_logger().setLevel('ERROR')
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

# Define training configuration

########################################################
# 1. batch size above 256 is not a good idea.
# 2. lr above 0.00028 is not a good idea.
# 3. adding more that 3 conv layers probably is not a good idea.
# 4. mae loss is not working.
# 5. lr scheduler is not making it better.
# 6. batch normalization is not improving the model based on the metrics, but it must be helping with the model robustness.
# 7. BN actually improved test_loss but not the test_mae. and also it affected the training and validation loss and mae negatively.
# 8. when using depthwise separable conv instead of normal conv, the performance degrades which says that we need to add complexitiy to the model to increase performance.
# 9. the more complex the model, the better the performance.
# 10. adding to much complexity is not a good idea, the model with 101k parameters is working better than the model with 331k parameters.
# 11. moving the activation functions after the add layers is not a good idea. lets keep them with the conv layers.

########################################################
config = {
    # Training parameters
    'learning_rate': 0.00028,  # Increased from 0.00014 to account for 4x larger batch size
    'batch_size': 128,
    'total_epochs': 1024,
    'early_stopping_patience': 30,
    'early_stopping_min_delta': 0.001,  # Minimum change in monitored metric to qualify as an improvement
    # Optimizer parameters
    'optimizer': 'sgd',
    'loss_function': 'mse',
    'performance_metrics': ['mae'],
    # Model checkpointing
    'save_best_only': True,
    'monitor_metric': 'val_loss'
}





def create_model():
    """Create a model for head pose estimation with skip connections using upsampling for dimension alignment."""
    # Base input layer
    inputs = keras.Input(shape=(None,None,96))
    
    # First layer - no skip connection
    x1 = keras.layers.Conv2D(
        filters=256,
        kernel_size=1,
        padding='same',
        activation='tanh',
        kernel_initializer=keras.initializers.GlorotUniform()
    )(inputs)
    
    x1 = keras.layers.Conv2D(
        filters=128,
        kernel_size=1,
        padding='same',
        activation='tanh',
        kernel_initializer=keras.initializers.GlorotUniform()
    )(x1)
    
    # Second layer - skip from input
    x2 = keras.layers.Conv2D(
        filters=64,
        kernel_size=1,
        padding='same',
        activation='tanh',
        kernel_initializer=keras.initializers.GlorotUniform()
    )(x1)
    
    # Skip connection from input to x2
    skip1 = keras.layers.Conv2D(
        filters=64,
        kernel_size=1,
        padding='same',
        activation='tanh',
        kernel_initializer=keras.initializers.GlorotUniform()
    )(inputs)
    x2 = keras.layers.Add()([x2, skip1])
    
    # Third layer - skip from x1
    x3 = keras.layers.Conv2D(
        filters=32,
        kernel_size=1,
        padding='same',
        activation='relu',
        kernel_initializer=keras.initializers.GlorotUniform()
    )(x2)
    
    # Skip connection from x1 to x3
    skip2 = keras.layers.Conv2D(
        filters=32,
        kernel_size=1,
        padding='same',
        activation='tanh',
        kernel_initializer=keras.initializers.GlorotUniform()
    )(x1)
    x3 = keras.layers.Add()([x3, skip2])
    
    # Fourth layer - skip from x2
    x4 = keras.layers.Conv2D(
        filters=16,
        kernel_size=1,
        padding='same',
        activation='tanh',
        kernel_initializer=keras.initializers.GlorotUniform()
    )(x3)
    
    # Skip connection from x2 to x4
    skip3 = keras.layers.Conv2D(
        filters=16,
        kernel_size=1,
        padding='same',
        activation='tanh',
        kernel_initializer=keras.initializers.GlorotUniform()
    )(x2)
    x4 = keras.layers.Add()([x4, skip3])
    
    # Final layer - no skip connection
    outputs = keras.layers.Conv2D(
        filters=3,
        kernel_size=1,
        padding='same',
        activation='linear',
        kernel_initializer=keras.initializers.GlorotUniform()
    )(x4)
    
    
    
    model = keras.Model(inputs=inputs, outputs=outputs)
    return model


def train():
    # Initialize wandb with config
    wandb.init(
        project="HeadPoseRegressor-BIWI-96features",
        config=config,
        notes="Restore the architecture of the model of the run za1bxuzn which was the best test mae with the lowest #params but train it this time only on BIWI_Train dataset.",
        tags=["BIWI_Train"]
    )
    
    # Load datasets
    print("Loading datasets...")
    train_features, train_poses = load_dataset('Head-Pose-Estimation-Model/FeatureMaps_Datasets/BIWI_train_features_96.npz')
    #train_features2, train_poses2 = load_dataset('processed_datasets/trainset2_features_96.npz')
    
    #train_features = np.concatenate((train_features, train_features2), axis=0)
    #train_poses = np.concatenate((train_poses, train_poses2), axis=0)
    
    test_features, test_poses = load_dataset('Head-Pose-Estimation-Model/FeatureMaps_Datasets/BIWI_test_features_96.npz')
    
    print(f"train_features shape: {train_features.shape}")
    print(f"train_poses shape: {train_poses.shape}")
    print(f"test_features shape: {test_features.shape}")
    print(f"test_poses shape: {test_poses.shape}")

    # Analyze angle distributions before training
    analyze_angle_distributions(train_poses, test_poses)

    # Reshape features
    train_features = train_features.reshape(-1, 1, 1, 96)
    test_features = test_features.reshape(-1, 1, 1, 96)

    # Split training data
    train_features, val_features, train_poses, val_poses = train_test_split(
        train_features, train_poses, 
        test_size=0.2, 
        random_state=42
    )

    # Create and compile model
    #model = create_model()
    model = load_model_from_json("Head-Pose-Estimation-Model/za1bxuzn.json")
    myoptimizer = keras.optimizers.SGD(learning_rate=config['learning_rate']) if config['optimizer'] == 'sgd' else keras.optimizers.Adam(learning_rate=config['learning_rate'])
    model.compile(
        optimizer=myoptimizer,
        loss=config['loss_function'],
        metrics=config['performance_metrics']
    )



    # Create callbacks
    callbacks = [
        keras.callbacks.ModelCheckpoint(
            f'Trained Models/model_runid_{wandb.run.id}.h5',
            monitor=config['monitor_metric'],
            save_best_only=config['save_best_only']
        ),
        keras.callbacks.EarlyStopping(
            monitor=config['monitor_metric'],
            patience=config['early_stopping_patience'],
            min_delta=config['early_stopping_min_delta'],
            restore_best_weights=True
        ),
        WandbCallback()
        # keras.callbacks.ReduceLROnPlateau(
        #     monitor='val_loss',
        #     factor=0.5,
        #     patience=10,
        #     min_lr=1e-6
        # )
    ]

    # Train the model
    history = model.fit(
        train_features,
        train_poses,
        epochs=config['total_epochs'],
        batch_size=config['batch_size'],
        validation_data=(val_features, val_poses),
        callbacks=callbacks,
        verbose=1
    )

    # Evaluate on test set
    test_loss, test_mae = model.evaluate(test_features, test_poses, verbose=2)

    # Create and log learning curves
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

    wandb.run.summary['test_loss'] = test_loss
    wandb.run.summary['test_mae'] = test_mae
    wandb.run.summary['total_parameters'] = model.count_params()
    wandb.run.summary['model_architecture'] = model.to_json()
    
    
    # Find best epoch based on validation loss
    best_epoch_idx = np.argmin(history.history['val_loss'])
    
    # Log best epoch and its metrics
    wandb.log({
        'best_epoch': best_epoch_idx + 1,  # Add 1 since epochs are 1-indexed
        'best_epoch_train_loss': history.history['loss'][best_epoch_idx],
        'best_epoch_train_mae': history.history['mae'][best_epoch_idx],
        'best_epoch_val_loss': history.history['val_loss'][best_epoch_idx], 
        'best_epoch_val_mae': history.history['val_mae'][best_epoch_idx]
    })

    

if __name__ == "__main__":
    train() 