import tensorflow as tf
from keras.models import load_model, Model
import argparse
import os
import numpy as np
import glob
import re
from tqdm import tqdm

''' 
The models that are for 96 dim features are mistakenly trained on input shape of 1*1*96 instead of None*None*96, which brings problem when we want to attach it to the 
BlazeFace model.
In this script, we convert the models input shape to None*None*96.
'''

def convert_input_shape(input_model_path, output_model_path):
    """
    Convert the input shape of a model from (batch, 1, 1, 96) to (batch, None, None, 96)
    and removes the Flatten layer, making the output come directly from the layer before it.
    
    Args:
        input_model_path (str): Path to the original model
        output_model_path (str): Path to save the converted model in .h5 format
    """
    # Check if input file exists
    if not os.path.exists(input_model_path):
        raise FileNotFoundError(f"Input model file {input_model_path} not found")
    
    # Ensure output path has .h5 extension
    if os.path.isdir(output_model_path):
        # If output_model_path is a directory, create a default filename
        model_filename = os.path.basename(input_model_path)
        model_name = os.path.splitext(model_filename)[0]
        output_model_path = os.path.join(output_model_path, f"{model_name}_converted.h5")
    elif not output_model_path.endswith('.h5'):
        output_model_path = f"{output_model_path}.h5"
    
    # Load the original model
    print(f"Loading model from {input_model_path}...")
    original_model = load_model(input_model_path)
    
    # Get the input shape of the model
    original_shape = original_model.input_shape
    print(f"Original input shape: {original_shape}")
    
    # Validate that the original shape has 96 channels
    if len(original_shape) != 4 or original_shape[3] != 96:
        raise ValueError(f"Expected input shape with 96 channels, got {original_shape}")
    
    # Find the Flatten layer
    flatten_layer = None
    for i, layer in enumerate(original_model.layers):
        if isinstance(layer, tf.keras.layers.Flatten) or layer.name.lower() == 'flatten':
            flatten_layer = layer
            flatten_index = i
            break
    
    if flatten_layer is None:
        print("Warning: No Flatten layer found in the model. Proceeding with full model conversion.")
        # Use the original approach for models without a Flatten layer
        model_config = original_model.get_config()
        
        # Temporarily save weights
        temp_weights = {}
        for layer in original_model.layers:
            if layer.weights:
                temp_weights[layer.name] = layer.get_weights()
        
        # Modify the input shape in the config
        input_layer_config = next(layer for layer in model_config['layers'] 
                                if layer['name'] == original_model.layers[0].name)
        input_layer_config['config']['batch_input_shape'] = (None, None, None, 96)
        
        # Create the new model from the modified config
        new_model = tf.keras.Model.from_config(model_config)
        
        # Restore weights
        for layer in new_model.layers:
            if layer.name in temp_weights:
                layer.set_weights(temp_weights[layer.name])
    else:
        print(f"Found Flatten layer: {flatten_layer.name} at index {flatten_index}")
        
        # Get the layer before the Flatten layer
        prev_layer = original_model.layers[flatten_index - 1]
        print(f"Using layer before Flatten as new output: {prev_layer.name}")
        
        # Create a new model from the input to the layer before Flatten
        input_layer = original_model.input
        output_layer = prev_layer.output
        
        # Create a modified model
        modified_model = tf.keras.Model(inputs=input_layer, outputs=output_layer)
        
        # Get the modified model config
        model_config = modified_model.get_config()
        
        # Temporarily save weights for the kept layers
        temp_weights = {}
        for layer in modified_model.layers:
            if layer.weights:
                temp_weights[layer.name] = layer.get_weights()
        
        # Modify the input shape in the config
        input_layer_config = next(layer for layer in model_config['layers'] 
                                if layer['name'] == modified_model.layers[0].name)
        input_layer_config['config']['batch_input_shape'] = (None, None, None, 96)
        
        # Create the new model from the modified config
        new_model = tf.keras.Model.from_config(model_config)
        
        # Restore weights
        for layer in new_model.layers:
            if layer.name in temp_weights:
                layer.set_weights(temp_weights[layer.name])
    
    # Verify the new input shape
    print(f"New input shape: {new_model.input_shape}")
    print(f"New output shape: {new_model.output_shape}")
    
    # Save the new model in h5 format
    print(f"Saving converted model to {output_model_path}...")
    new_model.save(output_model_path, save_format='h5')
    
    print(f"Model converted successfully!")
    
    return new_model

def validate_conversion(original_model_path, converted_model_path, num_samples=5):
    """
    Validates that the converted model produces identical outputs to the original model.
    Only works with models in .h5 format.
    
    Args:
        original_model_path (str): Path to the original model (.h5 format)
        converted_model_path (str): Path to the converted model (.h5 format) or directory
        num_samples (int): Number of random inputs to test
        
    Returns:
        bool: True if validation passes, False otherwise
    """
    print(f"\nValidating conversion...")
    
    # Check if original model file is .h5 format
    if not original_model_path.endswith('.h5'):
        raise ValueError(f"Original model must be in .h5 format: {original_model_path}")
    
    # If converted_model_path is a directory, derive the actual file path
    if os.path.isdir(converted_model_path):
        model_filename = os.path.basename(original_model_path)
        model_name = os.path.splitext(model_filename)[0]
        converted_model_path = os.path.join(converted_model_path, f"{model_name}_converted.h5")
        print(f"Output is a directory, using derived path: {converted_model_path}")
    elif not converted_model_path.endswith('.h5'):
        converted_model_path = f"{converted_model_path}.h5"
        print(f"Adding .h5 extension to output path: {converted_model_path}")
    
    # Verify files exist
    if not os.path.exists(original_model_path):
        raise FileNotFoundError(f"Original model file not found: {original_model_path}")
    if not os.path.exists(converted_model_path):
        raise FileNotFoundError(f"Converted model file not found: {converted_model_path}")
    
    # Load both models with h5 format explicitly
    print(f"Loading original model from: {original_model_path}")
    original_model = load_model(original_model_path, compile=False)
    
    print(f"Loading converted model from: {converted_model_path}")
    converted_model = load_model(converted_model_path, compile=False)
    
    # Get input shapes
    original_shape = original_model.input_shape
    converted_shape = converted_model.input_shape
    
    print(f"Original model input shape: {original_shape}")
    print(f"Converted model input shape: {converted_shape}")
    
    all_tests_passed = True
    
    for i in range(num_samples):
        print(f"\nTest {i+1}/{num_samples}:")
        
        # Create a random input for the original model
        # Assuming shape is (batch, height, width, channels)
        batch_size = 1
        height, width = original_shape[1:3]
        
        # If dimensions were originally 1x1, use those specific values
        if height == 1 and width == 1:
            original_input = np.random.rand(batch_size, height, width, 96).astype(np.float32)
            # For converted model, we can use the same shape since None dimensions accept any size
            converted_input = original_input
        else:
            # If the original model already had flexible dimensions
            # Generate random dimensions for testing
            height, width = 5, 5  # Example arbitrary dimensions
            original_input = np.random.rand(batch_size, height, width, 96).astype(np.float32)
            converted_input = original_input
        
        # Get predictions
        original_output = original_model.predict(original_input, verbose=0)
        converted_output = converted_model.predict(converted_input, verbose=0)
        
        # Compare outputs
        if np.allclose(original_output, converted_output, rtol=1e-5, atol=1e-5):
            print("✓ Outputs match!")
        else:
            print("✗ Outputs differ!")
            max_diff = np.max(np.abs(original_output - converted_output))
            print(f"  Maximum difference: {max_diff}")
            all_tests_passed = False
    
    if all_tests_passed:
        print("\n✅ Validation successful! Both models produce identical outputs.")
    else:
        print("\n❌ Validation failed! Models produce different outputs.")
    
    return all_tests_passed

def batch_convert_models(source_dir, dest_dir, num_validation_samples=10):
    """
    Converts input shape of all models in a directory and validates each conversion.
    
    Args:
        source_dir (str): Path to directory containing original models
        dest_dir (str): Path to directory where converted models will be saved
        num_validation_samples (int): Number of samples to use for validation
        
    Returns:
        dict: Summary of conversion results
    """
    # Ensure destination directory exists
    os.makedirs(dest_dir, exist_ok=True)
    
    # Find all .h5 files in the source directory
    model_files = glob.glob(os.path.join(source_dir, "model_runid_*.h5"))
    
    if not model_files:
        print(f"No models matching pattern 'model_runid_*.h5' found in {source_dir}")
        return {}
    
    print(f"Found {len(model_files)} models to convert")
    
    # Dictionary to store results
    results = {
        "total": len(model_files),
        "converted": 0,
        "validated": 0,
        "failed": 0,
        "details": {}
    }
    
    # Process each model file with progress bar
    for model_path in tqdm(model_files, desc="Converting models", colour="green"):
        # Extract the ID from the filename
        match = re.search(r'model_runid_([^.]+)\.h5', os.path.basename(model_path))
        if not match:
            print(f"Skipping {model_path}: filename doesn't match expected pattern")
            continue
            
        model_id = match.group(1)
        output_filename = f"{model_id}.h5"
        output_path = os.path.join(dest_dir, output_filename)
        
        try:
            # Convert model input shape
            print(f"\nProcessing model ID: {model_id}")
            convert_input_shape(model_path, output_path)
            results["converted"] += 1
            
            # Validate the conversion
            validation_result = validate_conversion(
                model_path, 
                output_path, 
                num_samples=num_validation_samples
            )
            
            # Store results
            results["details"][model_id] = {
                "original_path": model_path,
                "converted_path": output_path,
                "validation_passed": validation_result
            }
            
            if validation_result:
                results["validated"] += 1
            else:
                results["failed"] += 1
                
        except Exception as e:
            print(f"Error processing {model_path}: {str(e)}")
            results["details"][model_id] = {
                "original_path": model_path,
                "error": str(e)
            }
            results["failed"] += 1
    
    # Print summary
    print("\n=== Conversion Summary ===")
    print(f"Total models found: {results['total']}")
    print(f"Successfully converted: {results['converted']}")
    print(f"Validation passed: {results['validated']}")
    print(f"Failed: {results['failed']}")
    
    return results

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Convert model input shape from 1*1*96 to None*None*96')
    parser.add_argument('--input_model', help='Path to a single original model')
    parser.add_argument('--output_model', help='Path to save a single converted model')
    parser.add_argument('--source_dir', help='Directory containing multiple models to convert')
    parser.add_argument('--dest_dir', help='Directory to save multiple converted models')
    parser.add_argument('--validate', action='store_true', default=True, help='Validate the conversion')
    parser.add_argument('--samples', type=int, default=10, help='Number of validation samples')
    
    args = parser.parse_args()
    
    # Check if batch conversion is requested
    if args.source_dir and args.dest_dir:
        batch_convert_models(args.source_dir, args.dest_dir, args.samples)
    # # Single model conversion
    # elif args.input_model and args.output_model:
    #     convert_input_shape(args.input_model, args.output_model)
    #     if args.validate:
    #         validate_conversion(args.input_model, args.output_model, num_samples=args.samples)
    else:
        parser.print_help()