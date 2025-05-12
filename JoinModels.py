import tensorflow as tf
from keras.models import Model, load_model
import os

def join_models(face_detector_path, regressor1_path, regressor2_path, layer1_name, layer2_name, output_model_path, metadata: dict = None):
    """
    Join a face detector model with two regressor models.
    
    Args:       
        face_detector_path: Path to the face detector .h5 model
        regressor1_path: Path to the first regressor .h5 model
        regressor2_path: Path to the second regressor .h5 model
        layer1_name: Name of the layer in face detector where regressor1 should be attached
        layer2_name: Name of the layer in face detector where regressor2 should be attached
        output_model_path: Path where the unified model will be saved
        metadata: Optional metadata dictionary to store additional information about the model
    
    Returns:
        The unified model
    """
    
    # Check if the model files exist
    for path in [face_detector_path, regressor1_path, regressor2_path]:
        if not os.path.exists(path):
            raise FileNotFoundError(f"Model file not found: {path}")
    
    # Load the models
    print("Loading models...")
    face_detector = load_model(face_detector_path)
    regressor1 = load_model(regressor1_path)
    regressor2 = load_model(regressor2_path)
    
    # Check if the specified layers exist in the face detector
    face_detector_layer_names = [layer.name for layer in face_detector.layers]
    
    if layer1_name not in face_detector_layer_names:
        raise ValueError(f"Layer '{layer1_name}' not found in face detector model")
    
    if layer2_name not in face_detector_layer_names:
        raise ValueError(f"Layer '{layer2_name}' not found in face detector model")
    
    # Get the layer outputs from the face detector
    layer1_output = face_detector.get_layer(layer1_name).output
    layer2_output = face_detector.get_layer(layer2_name).output
    
    # Get the expected input shapes for the regressors
    regressor1_input_shape = regressor1.input_shape[1:]  # Skip batch dimension
    regressor2_input_shape = regressor2.input_shape[1:]  # Skip batch dimension
    
    print(f"Layer1 output shape: {layer1_output.shape[1:]}")
    print(f"Regressor1 input shape: {regressor1_input_shape}")
    print(f"Layer2 output shape: {layer2_output.shape[1:]}")
    print(f"Regressor2 input shape: {regressor2_input_shape}")
    
    # Check if the shapes are compatible and add reshape layers if needed
    if layer1_output.shape[1:] != regressor1_input_shape:
        print("Adding reshape layer for regressor1")
        layer1_output = tf.keras.layers.Reshape((16, 16, regressor1_input_shape[-1]))(layer1_output)

    if layer2_output.shape[1:] != regressor2_input_shape:
        print("Adding reshape layer for regressor2")
        layer2_output = tf.keras.layers.Reshape((8, 8 , regressor2_input_shape[-1]))(layer2_output)
    
    # Apply the regressors to the outputs from the face detector
    regressor1_output = regressor1(layer1_output)
    regressor2_output = regressor2(layer2_output)
    
    # Get all outputs from the face detector model
    face_detector_outputs = face_detector.outputs
    print(f"Face detector has {len(face_detector_outputs)} original outputs")
    
    # Create the unified model with all outputs: face detector outputs + regressor outputs
    all_outputs = face_detector_outputs + [regressor1_output, regressor2_output]
    
    unified_model = Model(
        inputs=face_detector.input,
        outputs=all_outputs
    )
    
    unified_model._metadata = metadata if metadata else {}
    
    # Print a summary of the unified model
    print("Unified model summary:")
    unified_model.summary()
   
    # Save the unified model
    unified_model.save(output_model_path)
    print(f"Unified model saved to {output_model_path}")
    
    return unified_model

def extract_id_from_path(file_path):
    """
    Extract the ID from a file path ending with "/{id}.h5".
    
    Args:
        file_path (str): The file path to extract the ID from.
    
    Returns:
        str: The extracted ID, or None if the path does not match the expected format.
    """
    # Ensure the file path ends with ".h5"
    if file_path.endswith(".h5"):
        # Extract the file name (last part of the path)
        file_name = os.path.basename(file_path)
        # Remove the ".h5" extension to get the ID
        return file_name[:-3]
    return None


if __name__ == "__main__":
    
    
    face_detector_path = "/Users/maaz/Desktop/ST-face-monitoring/Head-Pose-Estimation-Exp/BlazeFace-HeadPose/Get-BlazeFace-FeatureMaps-Dataset/BlazeFace/face_detection_front.h5"
    regressor1_path = "/Users/maaz/Desktop/ST-face-monitoring/Head-Pose-Estimation-Exp/BlazeFace-HeadPose/Head-Pose-Estimation-Model/Model-88/Trained-Models-88/stoqa9pt.h5"
    regressor2_path = "/Users/maaz/Desktop/ST-face-monitoring/Head-Pose-Estimation-Exp/BlazeFace-HeadPose/Head-Pose-Estimation-Model/Model-96/Trained-Models-96-ReshapedInput-NoFlatten/hrchr82r.h5"
    layer1_name ="re_lu_10"
    layer2_name = "re_lu_15"
    output_model_path = "/Users/maaz/Desktop/ST-face-monitoring/Head-Pose-Estimation-Exp/BlazeFace-HeadPose/Head-Pose-Estimation-Model/BlazePoser/UnifiedModels/"
    metadata = {"note": ""}
    
    
    modelid1 = extract_id_from_path(regressor1_path)
    modelid2 = extract_id_from_path(regressor2_path)
    output_model_path = os.path.join(output_model_path, f"reg1-{modelid1}-reg2-{modelid2}.h5")
    
    unified_model = join_models(
        face_detector_path,
        regressor1_path,
        regressor2_path,
        layer1_name,
        layer2_name,
        output_model_path,
        metadata
    )
    ###############################
    
    # UNIFIED_MODEL_PATH = "/Users/maaz/Desktop/ST-face-monitoring/Head-Pose-Estimation-Exp/Blaze-Face-Feature-Extractor/Head-Pose-Estimation-Model/UnifiedModels/reg1-stoqa9pt-reg2-cl4obelj.h5"
    
    # # Load the unified model
    # unified_model = load_model(UNIFIED_MODEL_PATH)
    # # generate a random input of range 0 to 255 with shape [1,128,128,3]
    # input_shape = unified_model.input_shape[1:]
    # random_input = tf.random.uniform(shape=(1, *input_shape), minval=0, maxval=255, dtype=tf.int32)
    # # Make predictions
    # predictions = unified_model.predict(random_input)
    # # Print the predictions 
    # print("Predictions:")
    # for i, pred in enumerate(predictions):
    #     print(f"Output {i}: {pred.shape}")

    # Predictions:
    # Output 0: (1, 512, 1)
    # Output 1: (1, 384, 1)
    # Output 2: (1, 512, 16)
    # Output 3: (1, 384, 16)
    # Output 4: (1, 16, 16, 3)
    # Output 5: (1, 8, 8, 3)
