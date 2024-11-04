from transformers import AutoConfig
import json
import os
from pprint import pprint
import argparse

def display_model_config(model_path):
    """
    Load and display model hyperparameters from config.json, including nested configurations
    for multimodal models
    
    Args:
        model_path (str): Path to the local model directory
    """
    try:
        # First check if config.json exists directly
        config_path = os.path.join(model_path, "config.json")
        if not os.path.exists(config_path):
            raise FileNotFoundError(f"Config file not found at {config_path}")
            
        # Method 1: Load raw config.json
        print("\nMethod 1: Raw config.json contents:")
        print("-" * 50)
        with open(config_path, 'r') as f:
            config_dict = json.load(f)
            pprint(config_dict)
            
        # Method 2: Load via AutoConfig
        print("\nMethod 2: Loaded via AutoConfig:")
        print("-" * 50)
        config = AutoConfig.from_pretrained(model_path, local_files_only=True)
        
        # Display basic model info without assumptions about specific attributes
        print("\nBasic Model Information:")
        print(f"Model Type: {config.model_type}")
        
        # Display all main configuration parameters
        print("\nMain Configuration Parameters:")
        for param, value in config.__dict__.items():
            # Skip private attributes, special methods, and nested configs (we'll handle them separately)
            if not param.startswith('_') and not isinstance(value, dict):
                print(f"{param}: {value}")
        
        # Handle nested configurations
        print("\nNested Configurations:")
        print("-" * 50)
        
        # Check for text_config
        if hasattr(config, 'text_config'):
            print("\nText Configuration:")
            text_config = config.text_config
            for param, value in text_config.__dict__.items():
                if not param.startswith('_') and not isinstance(value, dict):
                    print(f"text_config.{param}: {value}")
                
        # Check for vision_config
        if hasattr(config, 'vision_config'):
            print("\nVision Configuration:")
            vision_config = config.vision_config
            for param, value in vision_config.__dict__.items():
                if not param.startswith('_') and not isinstance(value, dict):
                    print(f"vision_config.{param}: {value}")
                
        # Check for any other nested configs in the raw dictionary
        for key, value in config_dict.items():
            if isinstance(value, dict) and key.endswith('_config') and key not in ['text_config', 'vision_config']:
                print(f"\n{key.title()}:")
                for param, val in value.items():
                    print(f"{key}.{param}: {val}")
        
        return config, config_dict
        
    except Exception as e:
        print(f"Error reading config: {str(e)}")
        raise

def main():
    # Set up argument parser
    parser = argparse.ArgumentParser(
        description='Display model configuration from a local HuggingFace model.'
    )
    parser.add_argument(
        'model_path',
        type=str,
        help='Path to the local model directory containing config.json'
    )
    parser.add_argument(
        '--raw-only',
        action='store_true',
        help='Only show raw config.json contents without parsing'
    )

    # Parse arguments
    args = parser.parse_args()

    try:
        if args.raw_only:
            # Only show raw config.json
            config_path = os.path.join(args.model_path, "config.json")
            with open(config_path, 'r') as f:
                config_dict = json.load(f)
                pprint(config_dict)
        else:
            # Show full config analysis
            config, raw_config = display_model_config(args.model_path)
            
    except Exception as e:
        print(f"Failed to read model config: {e}")
        return 1
    
    return 0

if __name__ == "__main__":
    exit(main())
