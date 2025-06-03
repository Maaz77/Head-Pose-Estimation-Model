import numpy as np
import tensorflow as tf
from tensorflow import keras
from sklearn.model_selection import train_test_split
import wandb
import os
import argparse
from utilities import WandbCallback, load_dataset, load_dataset_with_weights, load_model_from_json, analyze_angle_distributions
from dotenv import load_dotenv
load_dotenv()

# Suppress warnings
import warnings

warnings.filterwarnings('ignore')
tf.get_logger().setLevel('ERROR')
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'


RANDOM_SEED = 42
np.random.seed(RANDOM_SEED)
tf.random.set_seed(RANDOM_SEED)
os.environ['PYTHONHASHSEED'] = str(RANDOM_SEED)


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
    'learning_rate': 0.00028,  
    'batch_size': 128,
    'total_epochs': 10000,
    'early_stopping_patience': 40,
    'early_stopping_min_delta': 0.001,  
    # Optimizer parameters
    'optimizer': 'adam', 
    'loss_function': 'mse',
    'performance_metrics': ['mae'],
    # Model checkpointing
    'save_best_only': True,
    'monitor_metric': 'val_loss',
    'dropout_rate': -1,
    'regularizer_rate': -1, 
    'num_filters': -1
}





def create_model():
    
    regularizer = tf.keras.regularizers.l2(config['regularizer_rate'])

    
    inputs = keras.Input(shape=(None,None,96))
    
    x1 = keras.layers.Conv2D(
        filters=config['num_filters'],
        kernel_size=1,
        padding='same',
        activation='tanh',
        kernel_initializer=keras.initializers.GlorotUniform(),
        bias_regularizer=regularizer,
        kernel_regularizer=regularizer
    )(inputs)
    
    x1  = keras.layers.SpatialDropout2D(config['dropout_rate'])(x1)
    
    outputs = keras.layers.Conv2D(
        filters=3,  # 3 output values for pose angles
        kernel_size=1,
        padding='same',
        activation=None,  # No activation for regression output
        kernel_initializer=keras.initializers.GlorotUniform(),
        bias_regularizer=regularizer,
        kernel_regularizer=regularizer
    )(x1)
    
    outputs = keras.layers.SpatialDropout2D(config['dropout_rate'])(outputs)
    
    
    model = keras.Model(inputs=inputs, outputs=outputs)
    
    if config['optimizer'] == 'adamax':
        myoptimizer = keras.optimizers.Adamax(learning_rate=config['learning_rate'])
        
    else:
        myoptimizer = keras.optimizers.SGD(learning_rate=config['learning_rate']) if config['optimizer'] == 'sgd' else keras.optimizers.Adam(learning_rate=config['learning_rate'])
    
    model.compile(
        optimizer=myoptimizer,
        loss=config['loss_function'],
        metrics=config['performance_metrics']
    )
    return model


def train():
    # Initialize wandb with config
    wandb.init(
        project="HeadPoseRegressor-BIWI-96features",
        config=config,
        notes="",
        tags=["BIWI_Train"]
    )
    FEATUREMAPS_DIR_PATH = os.getenv('FEATUREMAPS_DIR_PATH')
    # Load datasets
    print("Loading datasets...")
    train_features, train_poses = load_dataset(f'{FEATUREMAPS_DIR_PATH}BIWI_train_features_96.npz')
    

    
    test_features, test_poses = load_dataset(f'{FEATUREMAPS_DIR_PATH}/BIWI_test_features_96.npz')
    
    test_AFLW200_features, test_AFLW2000_poses = load_dataset(f'{FEATUREMAPS_DIR_PATH}/AFLW2000_features_96_0.7_1.npz')
    


    train_features = train_features.reshape(-1, 1, 1, 96)
    test_features = test_features.reshape(-1, 1, 1, 96)
    test_AFLW200_features = test_AFLW200_features.reshape(-1, 1, 1, 96)
    
    train_poses = train_poses.reshape(-1, 1, 1, 3)
    test_poses = test_poses.reshape(-1, 1, 1, 3)
    test_AFLW2000_poses = test_AFLW2000_poses.reshape(-1, 1, 1, 3)

    train_features, val_features, train_poses, val_poses = train_test_split(
        train_features, train_poses, 
        test_size=0.2, 
        random_state=42
    )



    TRAINED_MODELS_96_RESHAPEDINPUT_NOFLATTEN_PATH = os.getenv('TRAINED_MODELS_96_RESHAPEDINPUT_NOFLATTEN_PATH')

    # Create callbacks
    callbacks = [
        keras.callbacks.ModelCheckpoint(
            f'{TRAINED_MODELS_96_RESHAPEDINPUT_NOFLATTEN_PATH}/{wandb.run.id}.h5',
            monitor=config['monitor_metric'],
            save_best_only=config['save_best_only']
        ),
        keras.callbacks.EarlyStopping(
            monitor=config['monitor_metric'],
            patience=config['early_stopping_patience'],
            min_delta=config['early_stopping_min_delta'],
            restore_best_weights=True
        ),
        WandbCallback(),
        # keras.callbacks.ReduceLROnPlateau(
        #     monitor='val_loss',
        #     factor=0.5,
        #     patience=10,
        #     min_lr=1e-6
        # )
    ]
    model = create_model()
    
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
    test_AFLW200_loss, test_AFLW200_mae = model.evaluate(test_AFLW200_features, test_AFLW2000_poses, verbose=2)



    wandb.run.summary['test_AFLW2000_mae'] = test_AFLW200_mae
    wandb.run.summary['test_AFLW2000_loss'] = test_AFLW200_loss
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
    
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--dropout_rate',
        type=float,
        default=config['dropout_rate'],
        
    )
    parser.add_argument(
        '--regularizer_rate',
        type=float,
        default=config['regularizer_rate'],
    )
    parser.add_argument(
        '--num_filters',
        type=int,
        default=config['num_filters'],
    )
    
    config.update(vars(parser.parse_args()))    

    
    train() 