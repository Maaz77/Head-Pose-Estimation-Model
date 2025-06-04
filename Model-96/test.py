import tensorflow as tf
import numpy as np
import wandb
import os
from dotenv import load_dotenv
load_dotenv()


def evaluate_head_pose_model(model_path, dataset_path):
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

def log_test_aflw2000_mae_for_runs(run_ids, model_base_path, dataset_path, project_name="HeadPoseRegressor-BIWI-96features"):
    """
    Log the test_AFLW2000_mae for a list of run IDs using wandb.
    
    Args:
        run_ids (list): List of run IDs to evaluate
        model_base_path (str): Base path where model files are stored
        dataset_path (str): Path to the AFLW2000 dataset (.npz file)
        project_name (str): The wandb project name
    
    Returns:
        dict: Dictionary mapping run_id to test_AFLW2000_mae values
    """
    api = wandb.Api()
    results = {}
    
    for run_id in run_ids:
        try:
            # Construct model path
            model_path = os.path.join(model_base_path, f"{run_id}.h5")
            
            # Check if model file exists
            if not os.path.exists(model_path):
                print(f"Model file not found for run_id {run_id}: {model_path}")
                results[run_id] = None
                continue
            
            # Evaluate model on AFLW2000 dataset
            print(f"Evaluating model for run_id: {run_id}")
            metrics = evaluate_head_pose_model(model_path, dataset_path)
            test_aflw2000_mae = metrics['MAE']['average']
            test_aflw2000_loss = metrics['MSE']['average']
            
            # Get the run from wandb
            run_path = f"abbaszadehmohammadamin-politecnico-di-milano/{project_name}/{run_id}"
            run = api.run(run_path)
            
            # Update the run with test_AFLW2000_mae and test_AFLW2000_loss
            run.summary["test_AFLW2000_mae"] = test_aflw2000_mae
            run.summary["test_AFLW2000_loss"] = test_aflw2000_loss
            run.update()
            
            # Store result
            results[run_id] = test_aflw2000_mae
            
            print(f"Run {run_id}: test_AFLW2000_mae = {test_aflw2000_mae:.4f}, test_AFLW2000_loss = {test_aflw2000_loss:.4f} - Logged to wandb")
            
        except Exception as e:
            print(f"Error processing run_id {run_id}: {str(e)}")
            results[run_id] = None
    
    return results

if __name__ == "__main__":
    pass

    #**********
    #**********
    #**********
    # FEATUREMAPS_DIR_PATH = os.getenv('FEATUREMAPS_DIR_PATH')
    
    # #load run IDs from the text file
    # run_ids_file_path = "/Users/maaz/Desktop/ST-face-monitoring/Head-Pose-Estimation-Exp/BlazeFace-HeadPose/Head-Pose-Estimation-Model/Model-96/missed_aflw200mae_runs.txt"

    # with open(run_ids_file_path, 'r') as file:
    #     run_ids = [line.strip() for line in file if line.strip()]

    

        
    # model_base_path = os.getenv('TRAINED_MODELS_96_RESHAPEDINPUT_NOFLATTEN_PATH')
    
    # results = log_test_aflw2000_mae_for_runs(
    #     run_ids=run_ids,
    #     model_base_path=model_base_path,
    #     dataset_path=f'{FEATUREMAPS_DIR_PATH}/AFLW2000_features_96_0.7_1.npz',
    #     project_name="HeadPoseRegressor-BIWI-96features"
    # )
    
    # print("\nSummary of results:")
    # for run_id, mae in results.items():
    #     if mae is not None:
    #         print(f"Run {run_id}: test_AFLW2000_mae = {mae:.4f}")
    #     else:
    #         print(f"Run {run_id}: Failed to evaluate")
    
    #**********
    #**********
    #**********
    
    
    

