import tensorflow as tf
from keras.utils import custom_object_scope
import h5py
import json
import numpy as np
import wandb
import os
from utilities import load_and_preprocess
from dotenv import load_dotenv
load_dotenv()



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
        'MAE': {angle_names[i]: float(mae_per_angle[i]) for i in range(3)},
        'MSE': {angle_names[i]: float(mse_per_angle[i]) for i in range(3)}
    }
    metrics['MAE']['average'] = float(np.mean(mae_per_angle))
    metrics['MSE']['average'] = float(np.mean(mse_per_angle))
    
    # Print metrics in a formatted way (similar to evaluate_head_pose_model_Reg)
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
    

def log_test_metrics_for_runs(run_ids, model_dir, dataset_path, project_name="HeadPoseRegressor-BIWI-96features", entity="abbaszadehmohammadamin-politecnico-di-milano"):
    """
    Evaluate models for given run IDs on AFLW2000 dataset and log test metrics to Weights & Biases.
    Uses MAE for test_AFLW2000_mae and MSE for test_AFLW2000_loss.
    
    Args:
        run_ids (list): List of run IDs to evaluate
        model_dir (str): Directory containing the model files (should contain .h5 files named with run IDs)
        dataset_path (str): Path to the AFLW2000 dataset file
        project_name (str): W&B project name
        entity (str): W&B entity name
        
    Returns:
        dict: Dictionary containing evaluation results for each run ID
    """
    api = wandb.Api()
    results = {}
    
    for run_id in run_ids:
        try:
            print(f"\nProcessing run ID: {run_id}")
            model_path = " "
            model_paths = [None, None]
            model_paths[0] = os.path.join(model_dir[0], f"{run_id}.h5")
            model_paths[1] = os.path.join(model_dir[1], f"{run_id}.h5")
            
            if os.path.exists(model_paths[0]):
                model_path = model_paths[0]
                
            elif os.path.exists(model_paths[1]):
                model_path = model_paths[1]
            else:
                raise FileNotFoundError(f"Model file for run ID {run_id} not found in specified directories.")
     
            
            # Evaluate model on AFLW2000 dataset
            print(f"Evaluating model: {model_path}")
            metrics = evaluate_head_pose_model_Reg_Cls(model_path, dataset_path)
            
            # Extract MAE and MSE metrics
            test_aflw2000_mae = metrics['MAE']['average']
            test_aflw2000_loss = metrics['MSE']['average']  # Using MSE as loss metric
            
            # Get the W&B run
            run_path = f"{entity}/{project_name}/{run_id}"
            run = api.run(run_path)
            
            # Log test metrics
            run.config["test_AFLW2000_mae"] = test_aflw2000_mae
            run.config["test_AFLW2000_loss"] = test_aflw2000_loss
            
            # Update the run
            run.update()
            
            # Store results
            results[run_id] = {
                'test_AFLW2000_mae': test_aflw2000_mae,
                'test_AFLW2000_loss': test_aflw2000_loss,
                'detailed_metrics': metrics
            }
            
            print(f"Successfully logged metrics for run {run_id}:")
            print(f"  test_AFLW2000_mae: {test_aflw2000_mae:.4f}")
            print(f"  test_AFLW2000_loss (MSE): {test_aflw2000_loss:.4f}")
            
        except Exception as e:
            print(f"Error processing run {run_id}: {str(e)}")
            results[run_id] = {'error': str(e)}
    
    return results


if __name__ == "__main__":
    
    ####*******
    #####*******

    # model_path1 = "Head-Pose-Estimation-Model/Trained Models/model_runid_rd93oeou.h5"
    # model_path2 = "Head-Pose-Estimation-Model/Trained Models/model_runid_wnfcrqss.h5"
    # model_path3 = "Head-Pose-Estimation-Model/Trained Models/model_runid_za1bxuzn.h5"
    # model_path4 = "Head-Pose-Estimation-Model/Trained Models/model_runid_cl4obelj.h5"
    # model_path5 = "Head-Pose-Estimation-Model/Trained Models/model_runid_yav3m4y3.h5"
    # model_path6 = "Trained Models/model_runid_bwhcoeb4.h5"
    # model_path7 = "/Users/maaz/Desktop/ST-face-monitoring/Head-Pose-Estimation-Exp/Blaze-Face-Feature-Extractor/Head-Pose-Estimation-Model/Model-96/Trained-Models-96-Reshaped-Input/rd93oeou.h5"
    # model_path8 = "/Users/maaz/Desktop/ST-face-monitoring/Head-Pose-Estimation-Exp/Blaze-Face-Feature-Extractor/Head-Pose-Estimation-Model/Model-96/Trained-Models-96-Reshaped-Input-NoFlatten/wnfcrqss.h5"
    # model_path9 = "/Users/maaz/Desktop/ST-face-monitoring/Head-Pose-Estimation-Exp/BlazeFace-HeadPose/Head-Pose-Estimation-Model/Model-96/Trained-Models-96-ReshapedInput-NoFlatten/hrchr82r.h5"
    
    # # Path to a Reg-Cls model
    # reg_cls_model_path = "/Users/maaz/Desktop/ST-face-monitoring/Head-Pose-Estimation-Exp/BlazeFace-HeadPose/Head-Pose-Estimation-Model/Model-96/Train-Models-96-Reg-Cls/0fi2hj25.h5"
    
    # dataset_path = "/Users/maaz/Desktop/ST-face-monitoring/Head-Pose-Estimation-Exp/BlazeFace-HeadPose/Head-Pose-Estimation-Model/FeatureMaps-Datasets/AFLW2000_features_96_0.7_1.npz"  
    
    # # Example: Single model evaluation
    # metrics = evaluate_head_pose_model_Reg_Cls(reg_cls_model_path, dataset_path)
    #####*******
    #####*******

    
    
    FEATUREMAPS_DIR_PATH = os.getenv('FEATUREMAPS_DIR_PATH')
    model_dir = []
    model_dir.append(os.getenv('TRAIN_MODEL_96_REG_CLS_PATH_NoHS'))
    model_dir.append(os.getenv('TRAIN_MODEL_96_REG_CLS_PATH'))

    # Read run IDs from the text file
    with open("/Users/maaz/Desktop/ST-face-monitoring/Head-Pose-Estimation-Exp/BlazeFace-HeadPose/Head-Pose-Estimation-Model/Model-96/runs_missed.txt", "r") as f:
        run_ids = [line.strip() for line in f if line.strip()]
            
    
    print("Logging test metrics for multiple runs...")
    results = log_test_metrics_for_runs(
        run_ids=run_ids,
        model_dir=model_dir,
        dataset_path=f'{FEATUREMAPS_DIR_PATH}/AFLW2000_features_96_0.7_1.npz',
        project_name="HeadPoseRegressor-BIWI-96features",
        entity="abbaszadehmohammadamin-politecnico-di-milano"
    )
    
    print("\nSummary of results:")
    for run_id, result in results.items():
        if 'error' in result:
            print(f"{run_id}: Error - {result['error']}")
        else:
            print(f"{run_id}: MAE={result['test_AFLW2000_mae']:.4f}, Loss={result['test_AFLW2000_loss']:.4f}")


