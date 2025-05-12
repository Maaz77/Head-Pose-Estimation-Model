import tensorflow as tf
import numpy as np

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
    
    
    dataset_path = "/Users/maaz/Desktop/ST-face-monitoring/Head-Pose-Estimation-Exp/BlazeFace-HeadPose/Head-Pose-Estimation-Model/FeatureMaps-Datasets/AFLW2000_features_96_0.7_1.npz"  
    
    metrics = evaluate_head_pose_model(model_path9, dataset_path)
    
    # print(f"\nYaw MAE: {metrics['MAE']['yaw']:.4f}")
    # print(f"Overall MSE: {metrics['MSE']['average']:.4f}")
    
