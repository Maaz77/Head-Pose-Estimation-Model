import numpy as np
import tensorflow as tf
from tensorflow import keras
from sklearn.model_selection import train_test_split
import wandb
import os
from datetime import datetime
import matplotlib.pyplot as plt
import io
from utilities import WandbCallback, load_dataset, load_model_from_json, analyze_angle_distributions, log_learningcurves
import warnings
from attention_model import se_transformer_regr_head , create_modelC , create_model_complex
from dotenv import load_dotenv
load_dotenv()  
 
 
warnings.filterwarnings('ignore')
tf.get_logger().setLevel('ERROR')
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

# Training Process Observations:

########################################################
#1. Starting from the simplest one conv layer with 3 filters, the complexer model gets the better results
#2. Using 'relu' for the middle layers activation does not improve the model.
#3. Using 'leaky_relu' for the activation of the middle conv layers is not better. 
#4. Using 'Relu' for last layer ruins the model.
# 5. removing the layer with 8 filters, a little improved the network. This layer deos not help to improve the model based on several experiments.
# 6. When increasing the batch size and learning rate, the model converges faster but the final accuracy on test set is lower. 
# 7. Increasing batchsize and lr, add more spikes or oscillation in the training curves.because the higher lr overshoots the optimal parameters and higher batch size cancle out the noise in the gradients which is good for smooth gradient descent.
# 8. Increasing model complexity by adding a conv layer with high(512) filters at the beginning of the netwrok, will help the network to converge faster during training but the test_mae is not improve. 
# 9. increasing lr and batchsize from 0.00028 and 128 does not improve the model performance.
# 10. adding more complexity like adding skip connections, converges faster but the test mae is not improved compared to intial stages simpler models.
# 11. when adding dropout the learning process stops sooner and the test mae is improved.
# 12. generally, adding model complexity and regularization or dropout improve the accuracy on test.
# 13. different activation functions are tested for middle layers. 
# 14. A Question is that if instead of regressing the labels use bins label and change to classification problem, will it help to improve the model performance?
# 15. I am witnessing that when the model is converging faster, the test mae is not improved on neither of the datasets.
    # For example these two runs: 94 and 93 
# 16. using kernel of 1*1 is beter than 3*3.
# 17. What I am witnessing is that using complex models is not beneficial. 
# 18. The more we increase the dataset and complexity we get better results.
# #######################################################

config = {
    # Training parameters
    'learning_rate': 0.00028,  # Increased from 0.00014 to account for 4x larger batch size
    'batch_size': 128 ,
    'total_epochs': 1000000,
    'early_stopping_patience': 40,
    'early_stopping_min_delta': 0.001,  # Minimum change in monitored metric to qualify as an improvement
    # Optimizer parameters
    'optimizer': 'sgd',
    'loss_function': 'mse',
    'performance_metrics': ['mae'],
    # Model checkpointing
    'save_best_only': True,
    'monitor_metric': 'val_loss',
    'dropout_rate' : 0.0001,
    'filtersnum' : 64,
    'regularizer_rate' : 1e-6   
}

def create_model():
    """Create a model for head pose estimation with skip connections using 1x1 convolutions for dimension alignment."""
    # Base input layer
    inputs = keras.Input(shape=(None, None, 88))
    
    # Regularizer
    regularizer = keras.regularizers.l2(config['regularizer_rate'])

    
    # First convolution block
    x0 = keras.layers.Conv2D(
        filters=config['filtersnum'],
        kernel_size=1,
        padding='same',
        activation='softsign',
        kernel_initializer=keras.initializers.GlorotUniform(),
        kernel_regularizer=regularizer
    )(inputs)
    x0 = keras.layers.SpatialDropout2D(config['dropout_rate'])(x0)
    
    # x1 = keras.layers.Conv2D(
    #     filters=config['filtersnum']/2,
    #     kernel_size=1,
    #     padding='same',
    #     activation='softsign',
    #     kernel_initializer=keras.initializers.GlorotUniform(),
    #     kernel_regularizer=regularizer
    # )(x0)
    # x1 = keras.layers.SpatialDropout2D(config['dropout_rate'])(x1)
    
    # # Second convolution block
    # x2 = keras.layers.Conv2D(
    #     filters=config['filtersnum']/4,
    #     kernel_size=1,
    #     padding='same',
    #     activation='softsign',
    #     kernel_initializer=keras.initializers.GlorotUniform(),
    #     kernel_regularizer=regularizer
    # )(x1)
    # x2 = keras.layers.SpatialDropout2D(config['dropout_rate'])(x2)
    
    # Third convolution block
    # x3 = keras.layers.Conv2D(
    #     filters=config['filtersnum']/4,
    #     kernel_size=1,
    #     padding='same',
    #     activation='softsign',
    #     kernel_initializer=keras.initializers.GlorotUniform(),
    #     kernel_regularizer=regularizer
    # )(x2)
    # x3 = keras.layers.SpatialDropout2D(config['dropout_rate'])(x3)
    
    # # --- Skip connection 1: from x1 to x3 ---
    # # Adjust x1 to match x3's number of filters
    # skip1 = keras.layers.Conv2D(
    #     filters= config['filtersnum']/4,
    #     kernel_size=1,
    #     padding='same',
    #     activation='softsign',
    #     kernel_initializer=keras.initializers.GlorotUniform(),
    #     kernel_regularizer=regularizer
    # )(x1)
    # skip1 = keras.layers.SpatialDropout2D(config['dropout_rate'])(skip1)
    # # Combine skip connection with x3 using element-wise addition
    # x3 = keras.layers.Add()([x3, skip1])
    
    # Fourth branch leading to output
    x5 = keras.layers.Conv2D(
        filters=3,
        kernel_size=1,
        padding='same',
        activation='linear',
        kernel_initializer=keras.initializers.GlorotUniform(),
        kernel_regularizer=regularizer
    )(x0)
    x5 = keras.layers.SpatialDropout2D(config['dropout_rate'])(x5)
    
    # # --- Skip connection 2: from x2 to output branch ---
    # # Adjust x2's filters to match x5's for addition
    # skip2 = keras.layers.Conv2D(
    #     filters=3,
    #     kernel_size=1,
    #     padding='same',
    #     activation='linear',
    #     kernel_initializer=keras.initializers.GlorotUniform(),
    #     kernel_regularizer=regularizer
    # )(x2)
    # skip2 = keras.layers.SpatialDropout2D(config['dropout_rate'])(skip2)
    # # Combine skip2 with x5
    # x5 = keras.layers.Add()([x5, skip2])
    
    model = keras.Model(inputs=inputs, outputs=x5)
    return model


# Assuming `config` dict with 'regularizer_rate' and 'dropout_rate'

def create_model_skip_fc():
    """
    Fully convolutional regressor with skip connections,
    spatial dropout, and L2 regularization.

    Input: (batch, H, W, 88)
    Output: (batch, H, W, 3)
    """
    regulizer = keras.regularizers.l2(config['regularizer_rate'])
    dr = config['dropout_rate']

    inputs = keras.Input(shape=(None, None, 88))  # dynamic H, W

    # --- Encoder block 1 ---
    x1 = keras.layers.Conv2D(
        filters=32,
        kernel_size=1,
        padding='same',
        activation='softsign',
        kernel_regularizer=regulizer,
        kernel_initializer='glorot_uniform'
    )(inputs)
    x1 = keras.layers.SpatialDropout2D(dr)(x1)

    # --- Encoder block 2 ---
    x2 = keras.layers.Conv2D(
        filters=64,
        kernel_size=1,
        padding='same',
        activation='softsign',
        kernel_regularizer=regulizer,
        kernel_initializer='glorot_uniform'
    )(x1)
    x2 = keras.layers.SpatialDropout2D(dr)(x2)

    # --- Decoder with skip connection ---
    x3 = keras.layers.Conv2D(
        filters=32,
        kernel_size=1,
        strides=1,
        padding='same',
        activation='softsign',
        kernel_regularizer=regulizer,
        kernel_initializer='glorot_uniform'
    )(x2)
    # Skip add from encoder block 1
    x3 = keras.layers.Add()([x3, x1])
    x3 = keras.layers.SpatialDropout2D(dr)(x3)

    # --- Final projection ---
    outputs = keras.layers.Conv2D(
        filters=3,
        kernel_size=1,
        padding='same',
        activation='linear',
        kernel_regularizer=regulizer,
        kernel_initializer='glorot_uniform'
    )(x3)

    model = keras.Model(inputs=inputs, outputs=outputs, name='FC_Skip_Regressor')
    return model


def bestmodelV1():
    """Create a model for head pose estimation with skip connections using upsampling for dimension alignment."""
    regulizer = keras.regularizers.l2(config['regularizer_rate'])
    # Base input layer
    inputs = keras.Input(shape=(None,None,88))
    
    x1 = keras.layers.Conv2D(
        filters= config['filtersnum'],
        kernel_size=1,
        padding='same',
        activation='softsign',
        kernel_regularizer=regulizer,
        kernel_initializer=keras.initializers.GlorotUniform()
    )(inputs)
    x1 = keras.layers.SpatialDropout2D(config['dropout_rate'])(x1)
    
    x2 = keras.layers.Conv2D(
        filters= 3,
        kernel_size=1,
        padding='same',
        activation='linear',
        kernel_regularizer=regulizer,
        kernel_initializer=keras.initializers.GlorotUniform()
    )(x1)
    x2 = keras.layers.SpatialDropout2D(config['dropout_rate'])(x2)
     
    model = keras.Model(inputs=inputs, outputs=x2)
    return model


def train():
    
    # Initialize wandb with config
    wandb.init(
        project="HeadPoseRegressor-88features",
        config=config,
        notes="",
        tags=["BIWI_Train+BIWI_NoTrack"] 
    )
    
    # Load datasets
    print("Loading datasets...")
    DIR_PATH = os.getenv('FEATUREMAPS_DIR_PATH')
    
    train_features, train_poses = load_dataset(DIR_PATH + 'BIWI_Train_Enlarged_features_88_0.7_1.npz')
    train_features2, train_poses2 = load_dataset(DIR_PATH + 'BIWI_NoTrack_Enlarged_features_88_0.7_1.npz')
    
    train_features = np.concatenate((train_features, train_features2), axis=0)
    train_poses = np.concatenate((train_poses, train_poses2), axis=0)
    
    test_features, test_poses = load_dataset(DIR_PATH + 'BIWI_Test_Enlarged_features_88_0.7_1.npz')
    
    
    test_features_AFLW2000, test_poses_AFLW2000 = load_dataset(DIR_PATH + 'AFLW2000_Enlarged_features_88_0.7_1.npz')
    
    print(f"train_features shape: {train_features.shape}")
    print(f"train_poses shape: {train_poses.shape}")
    print(f"test_features shape: {test_features.shape}")
    print(f"test_poses shape: {test_poses.shape}")

    # Analyze angle distributions before training
    analyze_angle_distributions(train_poses, test_poses)

    # Reshape features
    train_features = train_features.reshape(-1, 1, 1, 88)
    test_features = test_features.reshape(-1, 1, 1, 88)
    
    train_poses = train_poses.reshape(-1, 1, 1,3)
    test_poses = test_poses.reshape(-1, 1, 1,3)
    
    test_features_AFLW2000 = test_features_AFLW2000.reshape(-1, 1, 1, 88)
    test_poses_AFLW2000 = test_poses_AFLW2000.reshape(-1, 1, 1,3)
    

    # Split training data
    train_features, val_features, train_poses, val_poses = train_test_split(
        train_features, train_poses, 
        test_size=0.2, 
        random_state=42
    )

    # Create and compile model
    
    model = create_model_complex(config['regularizer_rate'], config['dropout_rate'])
    #model = create_modelC()
    #model = create_model()
    #model = create_model_skip_fc()
    #model = load_model_from_json("")
    #model = bestmodelV1()
    # model = se_transformer_regr_head(
    # input_channels=88,
    # reduction=4 ,
    # num_heads=1,
    # key_dim=8,
    # ff_dim=8,
    # hidden_channels=16
    # )
    myoptimizer = keras.optimizers.SGD(learning_rate=config['learning_rate']) if config['optimizer'] == 'sgd' else keras.optimizers.Adam(learning_rate=config['learning_rate'])
    model.compile(
        optimizer=myoptimizer,
        loss=config['loss_function'],
        metrics=config['performance_metrics']
    )



    # Create callbacks
    callbacks = [
        keras.callbacks.ModelCheckpoint(
            f'Trained-Models-88/{wandb.run.id}.h5',
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
    test_loss_biwi, test_mae_biwi = model.evaluate(test_features, test_poses, verbose=2)
    
    test_loss_AFLW2000, test_mae_AFLW2000 = model.evaluate(test_features_AFLW2000, test_poses_AFLW2000, verbose=2)
    
    print(f"Test loss on AFLW2000: {test_loss_AFLW2000}")
    print(f"Test MAE on AFLW2000: {test_mae_AFLW2000}")
    
    print(f"Test loss on BIWI_Test: {test_loss_biwi}")
    print(f"Test MAE on BIWI_Test: {test_mae_biwi}")

    
    

    wandb.run.summary['test_loss'] = test_loss_biwi
    wandb.run.summary['test_mae'] = test_mae_biwi
    wandb.run.summary['test_loss_AFLW2000'] = test_loss_AFLW2000
    wandb.run.summary['test_mae_AFLW2000'] = test_mae_AFLW2000
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
