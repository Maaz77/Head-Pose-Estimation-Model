import tensorflow as tf
from keras.utils import custom_object_scope
import h5py
import json
import numpy as np
from utilities import load_and_preprocess


def evaluate_head_pose_model_Reg(model_path, dataset_path):
    """
    Evaluate a head pose estimation model on a dataset and report MAE and MSE metrics.
    
    Args:
        model_path (str): Path to the .h5 model file
        dataset_path (str): Path to the .npz dataset file with 'features' and 'poses' keys
        
    Returns:
        dict: A dictionary containing MAE and MSE metrics for each angle and their averages
    """
    # Load the model
    model = tf.keras.models.load_model(model_path)
    
    # Load the dataset
    data = np.load(dataset_path)
    features = data['features']  # Shape: (batch_size, 96)
    ground_truth = data['poses']  # Shape: (batch_size, 3) [yaw, pitch, roll]
    
    batch_size = features.shape[0]
    
    # Reshape features to match model input shape (batch_size, 1, 1, 96)
    features_reshaped = features.reshape(batch_size, 1, 1, 96)
    
    # Make predictions
    predictions = model.predict(features_reshaped, verbose=0)
    
    # Ensure predictions have the expected shape
    if predictions.shape != ground_truth.shape:
        predictions = predictions.reshape(batch_size, 3)
    
    # Calculate metrics
    mae_per_angle = np.mean(np.abs(predictions - ground_truth), axis=0)
    mse_per_angle = np.mean(np.square(predictions - ground_truth), axis=0)
    
    avg_mae = np.mean(mae_per_angle)
    avg_mse = np.mean(mse_per_angle)
    
    # Prepare results
    angle_names = ['yaw', 'pitch', 'roll']
    metrics = {
        'MAE': {angle_names[i]: float(mae_per_angle[i]) for i in range(3)},
        'MSE': {angle_names[i]: float(mse_per_angle[i]) for i in range(3)},
    }
    metrics['MAE']['average'] = float(avg_mae)
    metrics['MSE']['average'] = float(avg_mse)
    
    # Print metrics in a formatted way
    print("Evaluation Results:")
    print("------------------")
    print("Mean Absolute Error (MAE):")
    for angle in angle_names:
        print(f"  {angle}: {metrics['MAE'][angle]:.4f}")
    print(f"  Average: {metrics['MAE']['average']:.4f}")
    
    print("\nMean Squared Error (MSE):")
    for angle in angle_names:
        print(f"  {angle}: {metrics['MSE'][angle]:.4f}")
    print(f"  Average: {metrics['MSE']['average']:.4f}")
    
    return metrics


def evaluate_head_pose_model_Reg_Cls(model_path, dataset_path):
    """
    Evaluate a head pose estimation model that performs both regression and classification on a dataset.
    Reports MAE metrics for the regression outputs only.
    
    Args:
        model_path (str): Path to the .h5 model file
        dataset_path (str): Path to the .npz dataset file with 'features' and 'poses' keys
        
    Returns:
        dict: A dictionary containing MAE metrics for each angle and their averages
    """
    # Configuration parameters (matching training setup)
    n_bins = 66
    M = 99.0
    

    model = tf.keras.models.load_model(model_path)
    
    # Load and preprocess the dataset to get both continuous poses and one-hot labels
    features, poses, weights, one_hot, bin_indices = load_and_preprocess(
        dataset_path, n_bins=n_bins, M=M
    )
    
    batch_size = features.shape[0]
    
    # Reshape features to match model input shape (batch_size, 1, 1, 96)
    features_reshaped = features.reshape(batch_size, 1, 1, 96)
    
    # Prepare ground truth targets in the format expected by the model
    ground_truth_dict = {
        'yaw_prob': one_hot[:, 0],      # Shape: (N, 66)
        'yaw_reg': poses[:, 0:1],       # Shape: (N, 1)
        'pitch_prob': one_hot[:, 1],    # Shape: (N, 66)
        'pitch_reg': poses[:, 1:2],     # Shape: (N, 1)
        'roll_prob': one_hot[:, 2],     # Shape: (N, 66)
        'roll_reg': poses[:, 2:3]       # Shape: (N, 1)
    }
    
    # Make predictions
    predictions = model.predict(features_reshaped, verbose=0)
    
    # Predictions are in order: [yaw_prob, yaw_reg, pitch_prob, pitch_reg, roll_prob, roll_reg]
    pred_yaw_prob = predictions[0]      # Shape: (N, 66)
    pred_yaw_reg = predictions[1]       # Shape: (N, 1)
    pred_pitch_prob = predictions[2]    # Shape: (N, 66)
    pred_pitch_reg = predictions[3]     # Shape: (N, 1)
    pred_roll_prob = predictions[4]     # Shape: (N, 66)
    pred_roll_reg = predictions[5]      # Shape: (N, 1)
    
    # Use model.evaluate to get the same metrics as in training
    evaluation_results = model.evaluate(
        x=features_reshaped,
        y=ground_truth_dict,
        sample_weight=weights,
        verbose=0
    )
    
    # Create metrics dictionary using model's metric names
    evaluation_metrics = {}
    for i, metric_name in enumerate(model.metrics_names):
        evaluation_metrics[metric_name] = float(evaluation_results[i])
    
    # Calculate additional custom metrics
    angle_names = ['yaw', 'pitch', 'roll']
    
    # Regression metrics (MAE and MSE)
    regression_metrics = {
        'MAE': {},
        'MSE': {}
    }
    
    # Calculate MAE and MSE for each angle using regression outputs
    reg_predictions = np.concatenate([pred_yaw_reg, pred_pitch_reg, pred_roll_reg], axis=1)  # Shape: (N, 3)
    ground_truth_poses = poses  # Shape: (N, 3)
    
    mae_per_angle = np.mean(np.abs(reg_predictions - ground_truth_poses), axis=0)
    mse_per_angle = np.mean(np.square(reg_predictions - ground_truth_poses), axis=0)
    
    for i, angle in enumerate(angle_names):
        regression_metrics['MAE'][angle] = float(mae_per_angle[i])
        regression_metrics['MSE'][angle] = float(mse_per_angle[i])
    
    regression_metrics['MAE']['average'] = float(np.mean(mae_per_angle))
    regression_metrics['MSE']['average'] = float(np.mean(mse_per_angle))
    
    # Classification metrics (accuracy)
    classification_metrics = {
        'accuracy': {}
    }
    
    # Calculate classification accuracy for each angle
    pred_yaw_class = np.argmax(pred_yaw_prob, axis=1)
    pred_pitch_class = np.argmax(pred_pitch_prob, axis=1)
    pred_roll_class = np.argmax(pred_roll_prob, axis=1)
    
    true_yaw_class = np.argmax(one_hot[:, 0], axis=1)
    true_pitch_class = np.argmax(one_hot[:, 1], axis=1)
    true_roll_class = np.argmax(one_hot[:, 2], axis=1)
    
    yaw_accuracy = np.mean(pred_yaw_class == true_yaw_class)
    pitch_accuracy = np.mean(pred_pitch_class == true_pitch_class)
    roll_accuracy = np.mean(pred_roll_class == true_roll_class)
    
    classification_metrics['accuracy']['yaw'] = float(yaw_accuracy)
    classification_metrics['accuracy']['pitch'] = float(pitch_accuracy)
    classification_metrics['accuracy']['roll'] = float(roll_accuracy)
    classification_metrics['accuracy']['average'] = float(np.mean([yaw_accuracy, pitch_accuracy, roll_accuracy]))

    
    # Create simplified metrics dict matching evaluate_head_pose_model_Reg format
    metrics = {
        'MAE': {angle_names[i]: float(mae_per_angle[i]) for i in range(3)}
    }
    metrics['MAE']['average'] = float(np.mean(mae_per_angle))
    
    # Print metrics in a formatted way (similar to evaluate_head_pose_model_Reg)
    print("Evaluation Results:")
    print("------------------")
    print("Mean Absolute Error (MAE):")
    for angle in angle_names:
        print(f"  {angle}: {metrics['MAE'][angle]:.4f}")
    print(f"  Average: {metrics['MAE']['average']:.4f}")
    
    return metrics
    

if __name__ == "__main__":
    
    model_path1 = "Head-Pose-Estimation-Model/Trained Models/model_runid_rd93oeou.h5"
    model_path2 = "Head-Pose-Estimation-Model/Trained Models/model_runid_wnfcrqss.h5"
    model_path3 = "Head-Pose-Estimation-Model/Trained Models/model_runid_za1bxuzn.h5"
    model_path4 = "Head-Pose-Estimation-Model/Trained Models/model_runid_cl4obelj.h5"
    model_path5 = "Head-Pose-Estimation-Model/Trained Models/model_runid_yav3m4y3.h5"
    model_path6 = "Trained Models/model_runid_bwhcoeb4.h5"
    model_path7 = "/Users/maaz/Desktop/ST-face-monitoring/Head-Pose-Estimation-Exp/Blaze-Face-Feature-Extractor/Head-Pose-Estimation-Model/Model-96/Trained-Models-96-Reshaped-Input/rd93oeou.h5"
    model_path8 = "/Users/maaz/Desktop/ST-face-monitoring/Head-Pose-Estimation-Exp/Blaze-Face-Feature-Extractor/Head-Pose-Estimation-Model/Model-96/Trained-Models-96-Reshaped-Input-NoFlatten/wnfcrqss.h5"
    model_path9 = "/Users/maaz/Desktop/ST-face-monitoring/Head-Pose-Estimation-Exp/BlazeFace-HeadPose/Head-Pose-Estimation-Model/Model-96/Trained-Models-96-ReshapedInput-NoFlatten/hrchr82r.h5"
    
    # Path to a Reg-Cls model
    reg_cls_model_path = "/Users/maaz/Desktop/ST-face-monitoring/Head-Pose-Estimation-Exp/BlazeFace-HeadPose/Head-Pose-Estimation-Model/Model-96/Train-Models-96-Reg-Cls/0fi2hj25.h5"
    
    dataset_path = "/Users/maaz/Desktop/ST-face-monitoring/Head-Pose-Estimation-Exp/BlazeFace-HeadPose/Head-Pose-Estimation-Model/FeatureMaps-Datasets/AFLW2000_features_96_0.7_1.npz"  
    
    metrics = evaluate_head_pose_model_Reg_Cls(reg_cls_model_path, dataset_path)
    

    
    # Uncomment to test regression-only model for comparison
    # print("\n" + "="*50)
    # print("Testing Reg-only model for comparison:")
    # reg_metrics = evaluate_head_pose_model_Reg(model_path9, dataset_path)
    
