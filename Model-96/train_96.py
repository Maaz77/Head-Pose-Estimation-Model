import numpy as np
import tensorflow as tf
from sklearn.model_selection import train_test_split
import wandb
import os
import keras
import matplotlib.pyplot as plt
from utilities import WandbCallback, load_dataset, load_and_preprocess, analyze_angle_distributions
from keras import layers, models, losses , activations
# Suppress warnings

import warnings

warnings.filterwarnings('ignore')
tf.get_logger().setLevel('ERROR')
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

from dotenv import load_dotenv
load_dotenv()  

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
    'total_epochs': 5000,
    'early_stopping_patience': 40,
    'early_stopping_min_delta': 0.001,  # Minimum change in monitored metric to qualify as an improvement
    # Optimizer parameters
    'optimizer': 'adamax', 
    'loss_function': 'mse+crossentropy',
    'performance_metrics': ['mae'],
    # Model checkpointing
    'save_best_only': True,
    'monitor_metric': 'val_loss',
    'dropout_rate' : 0.0001,
    'regularizer_rate' : 1e-2, 
    'n_bins': 66,   
    'M': 99.0       # Maximum angle in degrees
}


def build_model( 
    input_dim : int = 96,
    n_bins: int = config['n_bins'],
    M: float = config['M']
):
    """
    Build a Keras model with:
      - shared CNN backbone
      - 3 softmax classification heads (yaw, pitch, roll)
      - 3 regression heads via expectation over bin midpoints
    """
    # Create regularizer from config
    regularizer = tf.keras.regularizers.l2(config['regularizer_rate'])
    
    # Define hard swish activation function
    def hard_swish(x):
        return x * tf.nn.relu6(x + 3) / 6
    
    inp = layers.Input(shape=(None, None, input_dim), name='input_image')

    # --- backbone ---
    x = layers.Conv2D(10, 1, activation=hard_swish, padding='same', 
                     kernel_regularizer=regularizer, bias_regularizer=regularizer)(inp)
    x = layers.SpatialDropout2D(config['dropout_rate'])(x)
    x = layers.GlobalAveragePooling2D()(x)

    # --- classification logits ---
    yaw_logits   = layers.Dense(n_bins, name='yaw_cls', 
                              kernel_regularizer=regularizer, bias_regularizer=regularizer)(x)
    pitch_logits = layers.Dense(n_bins, name='pitch_cls', 
                              kernel_regularizer=regularizer, bias_regularizer=regularizer)(x)
    roll_logits  = layers.Dense(n_bins, name='roll_cls', 
                              kernel_regularizer=regularizer, bias_regularizer=regularizer)(x)

    yaw_prob   = layers.Activation('softmax', name='yaw_prob')(yaw_logits)
    pitch_prob = layers.Activation('softmax', name='pitch_prob')(pitch_logits)
    roll_prob  = layers.Activation('softmax', name='roll_prob')(roll_logits)

    # --- regression via expectation ---
    # compute bin midpoints of [-M, M]
    delta = 2 * M / n_bins
    midpoints = np.linspace(-M + delta/2, M - delta/2, n_bins).astype(np.float32)
    


    def expectation(p):
        # Use NumPy array directly instead of TensorFlow constant
        return tf.reduce_sum(p * midpoints, axis=-1, keepdims=True)

    yaw_reg   = layers.Lambda(expectation, name='yaw_reg')(yaw_prob)
    pitch_reg = layers.Lambda(expectation, name='pitch_reg')(pitch_prob)
    roll_reg  = layers.Lambda(expectation, name='roll_reg')(roll_prob)

    model = models.Model(
        inputs = inp,
        outputs= [yaw_prob,   yaw_reg,
                  pitch_prob, pitch_reg,
                  roll_prob,  roll_reg]
    )

    # compile with hybrid loss
    loss_fns = {
        'yaw_prob':   losses.CategoricalCrossentropy(),
        'yaw_reg':    losses.MeanSquaredError(),
        'pitch_prob': losses.CategoricalCrossentropy(),
        'pitch_reg':  losses.MeanSquaredError(),
        'roll_prob':  losses.CategoricalCrossentropy(),
        'roll_reg':   losses.MeanSquaredError(),
    }
    
    scheduler = keras.optimizers.schedules.ExponentialDecay(
        initial_learning_rate=config['learning_rate'],
        decay_steps=5000,
        decay_rate=0.96,
        staircase=True
    )
    
    # Create optimizer with learning rate from config
    if config['optimizer'].lower().strip() == 'adam':
        optimizer = tf.keras.optimizers.legacy.Adam(learning_rate=config['learning_rate'])
    elif config['optimizer'].lower().strip() == 'adadelta':
        optimizer = tf.keras.optimizers.legacy.Adadelta(learning_rate=config['learning_rate'])
    elif config['optimizer'].lower().strip() == 'sgd':
        optimizer = tf.keras.optimizers.legacy.SGD(learning_rate=config['learning_rate'])
    elif config['optimizer'].lower().strip() == 'rmsprop':
        optimizer = tf.keras.optimizers.legacy.RMSprop(learning_rate=config['learning_rate'])
    elif config['optimizer'].lower().strip() == 'adamax':
        optimizer = tf.keras.optimizers.legacy.Adamax(learning_rate=config['learning_rate'])
    elif config['optimizer'].lower().strip() == 'adagrad':
        optimizer = tf.keras.optimizers.legacy.Adagrad(learning_rate=config['learning_rate'])
    elif config['optimizer'].lower().strip() == 'nadam':
        optimizer = tf.keras.optimizers.legacy.Nadam(learning_rate=config['learning_rate'])
    elif config['optimizer'].lower().strip() == 'ftrl':
        optimizer = tf.keras.optimizers.legacy.Ftrl(learning_rate=config['learning_rate'])
    else: 
        raise ValueError(f"Unsupported optimizer: {config['optimizer']}")
    
    model.compile(
        optimizer=optimizer,
        loss=loss_fns,
        metrics={
            'yaw_reg': 'mae',
            'pitch_reg': 'mae',
            'roll_reg': 'mae',
            'yaw_prob': None,
            'pitch_prob': None,
            'roll_prob': None
    }
            
    )
    
    return model




def train():
    
    wandb.init(
        project="HeadPoseRegressor-BIWI-96features",
        config=config,
        notes="",
        tags=["BIWI_Train", "Reg+CLS_Loss", "Weighted_Samples"] # Available tags: "BIWI_Train", "Reg+CLS_Loss", "Weighted_Samples", "BIWInoTrack_And_BIWITrain"
    )
    
    print("Loading datasets...")
    DIR_PATH = os.getenv('FEATUREMAPS_DIR_PATH')
    train_features, train_poses, train_weights, train_one_hot, _ = load_and_preprocess(DIR_PATH + 'BIWI_train_features_96.npz', n_bins=config['n_bins'])
    
    train_features = train_features.reshape(-1, 1, 1, 96)
    
    print(f"train_features shape: {train_features.shape}")
    print(f"train_poses shape: {train_poses.shape}")
    print(f"train_weights shape: {train_weights.shape}")
    print(f"train_one_hot shape: {train_one_hot.shape}")
    
    train_features
    
    test_features, test_poses, test_weights, test_one_hot, _ = load_and_preprocess(DIR_PATH + 'BIWI_test_features_96.npz' , n_bins=config['n_bins'])
    
    test_features = test_features.reshape(-1, 1, 1, 96)
    
    print(f"test_features shape: {test_features.shape}")
    print(f"test_poses shape: {test_poses.shape}")
    print(f"test_weights shape: {test_weights.shape}")
    print(f"test_one_hot shape: {test_one_hot.shape}")
    
    
    
    train_features, val_features, train_poses, val_poses, train_weights, val_weights, train_one_hot, val_one_hot = train_test_split(
        train_features, train_poses, train_weights, train_one_hot,
        test_size=0.2, 
        random_state=42
    )
    
    mymodel = build_model()
    
    
    train_yaw_prob = train_one_hot[:, 0]    # Shape: (N, 66)
    train_pitch_prob = train_one_hot[:, 1]  # Shape: (N, 66)
    train_roll_prob = train_one_hot[:, 2]   # Shape: (N, 66)
    
    train_yaw_reg = train_poses[:, 0:1]     # Shape: (N, 1)
    train_pitch_reg = train_poses[:, 1:2]   # Shape: (N, 1)
    train_roll_reg = train_poses[:, 2:3]    # Shape: (N, 1)
    
    # Validation outputs
    val_yaw_prob = val_one_hot[:, 0]
    val_pitch_prob = val_one_hot[:, 1]
    val_roll_prob = val_one_hot[:, 2]
    
    val_yaw_reg = val_poses[:, 0:1]
    val_pitch_reg = val_poses[:, 1:2]
    val_roll_reg = val_poses[:, 2:3]
    
    # Test outputs
    test_yaw_prob = test_one_hot[:, 0]
    test_pitch_prob = test_one_hot[:, 1]
    test_roll_prob = test_one_hot[:, 2]
    
    test_yaw_reg = test_poses[:, 0:1]
    test_pitch_reg = test_poses[:, 1:2]
    test_roll_reg = test_poses[:, 2:3]
    
    TRAINED_MODELS_PATH = os.getenv('TRAIN_MODEL_96_REG_CLS_PATH')

    callbacks = [
        
        

        # keras.callbacks.ReduceLROnPlateau(
        #     monitor=config['monitor_metric'],
        #     factor=0.5,             # Halves the LR each time
        #     patience=20,            # Waits 20 epochs after val_loss stops improving
        #     min_delta=1e-4,         # Slight improvement required to reset patience
        #     cooldown=5,             # Waits 5 epochs after LR drop before monitoring again
        #     min_lr=1e-6,            # Don’t reduce LR below this
        #     verbose=1
        # ),
        

        
        keras.callbacks.ModelCheckpoint(
            f'{TRAINED_MODELS_PATH}/{wandb.run.id}.h5',
            monitor=config['monitor_metric'],
            save_best_only=config['save_best_only']
        ),
        keras.callbacks.EarlyStopping(
            monitor=config['monitor_metric'],
            patience=config['early_stopping_patience'],
            min_delta=config['early_stopping_min_delta'],
            restore_best_weights=config['save_best_only']
        ),
        WandbCallback()
    ]
    
    train_y = {
        'yaw_prob': train_yaw_prob,
        'yaw_reg': train_yaw_reg,
        'pitch_prob': train_pitch_prob,
        'pitch_reg': train_pitch_reg,
        'roll_prob': train_roll_prob,
        'roll_reg': train_roll_reg
    }
    
    val_y = {
        'yaw_prob': val_yaw_prob,
        'yaw_reg': val_yaw_reg,
        'pitch_prob': val_pitch_prob,
        'pitch_reg': val_pitch_reg,
        'roll_prob': val_roll_prob,
        'roll_reg': val_roll_reg
    }
    
    history = mymodel.fit(
        x=train_features,
        y=train_y,
        epochs=config['total_epochs'],
        batch_size=config['batch_size'],
        validation_data=(val_features, val_y),
        sample_weight=train_weights,
        callbacks=callbacks,
        verbose=2
    )
    
    # Evaluate on test set
    test_y = {
        'yaw_prob': test_yaw_prob,
        'yaw_reg': test_yaw_reg,
        'pitch_prob': test_pitch_prob,
        'pitch_reg': test_pitch_reg,
        'roll_prob': test_roll_prob,
        'roll_reg': test_roll_reg
    }
    
    test_results = mymodel.evaluate(
        x=test_features,
        y=test_y,
        sample_weight=test_weights,
        verbose=0
    )
    
    # Log test results to wandb
    test_metrics = {}
    for i, metric_name in enumerate(mymodel.metrics_names):
        test_metrics[f'test_{metric_name}'] = test_results[i]
    
    # Calculate mean regression loss for test data
    mean_test_reg_loss = np.mean([
        test_metrics['test_yaw_reg_loss'],
        test_metrics['test_pitch_reg_loss'],
        test_metrics['test_roll_reg_loss']
    ])
    
    # Calculate mean MAE for test data
    mean_test_mae = np.mean([
        test_metrics['test_yaw_reg_mae'],
        test_metrics['test_pitch_reg_mae'],
        test_metrics['test_roll_reg_mae']
    ])
            
    wandb.run.summary['total_parameters'] = mymodel.count_params()
    wandb.run.summary['test_loss'] = mean_test_reg_loss
    wandb.run.summary['test_mae'] = mean_test_mae
    wandb.run.summary['model_architecture'] = mymodel.to_json()



    
if __name__ == "__main__":
    
    train()


