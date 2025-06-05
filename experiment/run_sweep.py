#!/usr/bin/env python3
"""
Automated sweep script to train and validate RWKV models with different D_MODEL and NUM_SAMPLES configurations.
This script will:
1. Generate dataset with each NUM_SAMPLES value
2. Train a model with each D_MODEL value for each dataset
3. Automatically validate the trained model
4. Move to the next configuration
"""

import os
import shutil
import subprocess
import sys
import time
from pathlib import Path

# Add the current directory to Python path
current_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, current_dir)

# Import config to modify it
from config import MODEL_CHECKPOINT_PATH_CONFIG, MODEL_HYPERPARAMETERS

# Sweep configuration
D_MODEL_VALUES = [10, 20]
NUM_SAMPLES_VALUES = [100, 200]  # Add this new parameter
BACKUP_ORIGINAL_CONFIG = True
CLEANUP_MODELS = False  # Set to True if you want to delete model files after validation

def backup_config():
    """Backup the original config file"""
    if BACKUP_ORIGINAL_CONFIG:
        config_path = os.path.join(current_dir, 'config.py')
        backup_path = os.path.join(current_dir, 'config_original_backup.py')
        if not os.path.exists(backup_path):
            shutil.copy2(config_path, backup_path)
            print(f"✓ Original config backed up to: {backup_path}")

def update_config(d_model_value, num_samples_value):
    """Update the config.py file with new D_MODEL and NUM_SAMPLES values"""
    config_path = os.path.join(current_dir, 'config.py')
    
    # Read the current config
    with open(config_path, 'r') as f:
        content = f.read()
    
    # Replace the D_MODEL value
    import re
    d_model_pattern = r'"D_MODEL":\s*\d+,'
    d_model_replacement = f'"D_MODEL": {d_model_value},'
    content = re.sub(d_model_pattern, d_model_replacement, content)
    
    # Replace the NUM_SAMPLES value
    num_samples_pattern = r'"NUM_SAMPLES":\s*\d+,'
    num_samples_replacement = f'"NUM_SAMPLES": {num_samples_value},'
    content = re.sub(num_samples_pattern, num_samples_replacement, content)
    
    # Write back the updated config
    with open(config_path, 'w') as f:
        f.write(content)
    
    print(f"✓ Updated config.py: D_MODEL = {d_model_value}, NUM_SAMPLES = {num_samples_value}")

def run_dataset_generation():
    """Run the dataset generation script"""
    print("\n" + "="*60)
    print("📊 STARTING DATASET GENERATION")
    print("="*60)
    
    try:
        # Run dataset generation script
        result = subprocess.run([
            sys.executable, 'dataset_generator.py'
        ], cwd=current_dir, capture_output=False, text=True)
        
        if result.returncode == 0:
            print("✓ Dataset generation completed successfully")
            return True
        else:
            print(f"❌ Dataset generation failed with return code: {result.returncode}")
            return False
            
    except Exception as e:
        print(f"❌ Dataset generation failed with exception: {e}")
        return False

def run_training():
    """Run the training script"""
    print("\n" + "="*60)
    print("🚀 STARTING TRAINING")
    print("="*60)
    
    try:
        # Run training script
        result = subprocess.run([
            sys.executable, 'train_rwkv_regex.py'
        ], cwd=current_dir, capture_output=False, text=True)
        
        if result.returncode == 0:
            print("✓ Training completed successfully")
            return True
        else:
            print(f"❌ Training failed with return code: {result.returncode}")
            return False
            
    except Exception as e:
        print(f"❌ Training failed with exception: {e}")
        return False

def run_validation():
    """Run the validation script"""
    print("\n" + "="*60)
    print("🔍 STARTING VALIDATION")
    print("="*60)
    
    try:
        # Run validation script
        result = subprocess.run([
            sys.executable, 'validate_model_on_txt.py'
        ], cwd=current_dir, capture_output=False, text=True)
        
        if result.returncode == 0:
            print("✓ Validation completed successfully")
            return True
        else:
            print(f"❌ Validation failed with return code: {result.returncode}")
            return False
            
    except Exception as e:
        print(f"❌ Validation failed with exception: {e}")
        return False

def backup_model_file(d_model_value, num_samples_value):
    """Backup the trained model file with D_MODEL and NUM_SAMPLES suffix"""
    model_file = os.path.join(current_dir, MODEL_CHECKPOINT_PATH_CONFIG)
    if os.path.exists(model_file):
        backup_file = os.path.join(current_dir, f"rwkv7_fsm_experimental_model_d{d_model_value}_s{num_samples_value}.pth")
        shutil.copy2(model_file, backup_file)
        print(f"✓ Model backed up as: {backup_file}")
        return backup_file
    else:
        print(f"⚠️ Model file not found: {model_file}")
        return None

def cleanup_model_file():
    """Remove the current model file to save space"""
    if CLEANUP_MODELS:
        model_file = os.path.join(current_dir, MODEL_CHECKPOINT_PATH_CONFIG)
        if os.path.exists(model_file):
            os.remove(model_file)
            print(f"🗑️ Cleaned up model file: {model_file}")

def restore_original_config():
    """Restore the original config file"""
    if BACKUP_ORIGINAL_CONFIG:
        config_path = os.path.join(current_dir, 'config.py')
        backup_path = os.path.join(current_dir, 'config_original_backup.py')
        if os.path.exists(backup_path):
            shutil.copy2(backup_path, config_path)
            print(f"✓ Original config restored from: {backup_path}")

def main():
    print("🎯 RWKV Model D_MODEL & NUM_SAMPLES Sweep Script")
    print("="*80)
    print(f"Will train and validate models with:")
    print(f"  D_MODEL values: {D_MODEL_VALUES}")
    print(f"  NUM_SAMPLES values: {NUM_SAMPLES_VALUES}")
    print(f"  Total configurations: {len(D_MODEL_VALUES) * len(NUM_SAMPLES_VALUES)}")
    print(f"Current directory: {current_dir}")
    
    # Backup original config
    backup_config()
    
    successful_runs = []
    failed_runs = []
    
    total_start_time = time.time()
    config_count = 0
    total_configs = len(D_MODEL_VALUES) * len(NUM_SAMPLES_VALUES)
    
    for num_samples_value in NUM_SAMPLES_VALUES:
        print(f"\n{'🔢'*30}")
        print(f"DATASET CONFIGURATION: NUM_SAMPLES = {num_samples_value}")
        print(f"{'🔢'*30}")
        
        # Generate dataset for this NUM_SAMPLES value
        # First update config with any D_MODEL (we'll change it later) and the NUM_SAMPLES
        update_config(D_MODEL_VALUES[0], num_samples_value)
        
        dataset_success = run_dataset_generation()
        if not dataset_success:
            print(f"❌ Dataset generation failed for NUM_SAMPLES={num_samples_value}")
            # Mark all D_MODEL combinations for this NUM_SAMPLES as failed
            for d_model_value in D_MODEL_VALUES:
                failed_runs.append(f"D_MODEL={d_model_value}_NUM_SAMPLES={num_samples_value} (Dataset)")
            continue
        
        # Now train models with different D_MODEL values using this dataset
        for d_model_value in D_MODEL_VALUES:
            config_count += 1
            print(f"\n{'🔄'*25}")
            print(f"CONFIGURATION {config_count}/{total_configs}: D_MODEL = {d_model_value}, NUM_SAMPLES = {num_samples_value}")
            print(f"{'🔄'*25}")
            
            run_start_time = time.time()
            
            try:
                # Update config with this D_MODEL (NUM_SAMPLES already set)
                update_config(d_model_value, num_samples_value)
                
                # Run training
                training_success = run_training()
                if not training_success:
                    failed_runs.append(f"D_MODEL={d_model_value}_NUM_SAMPLES={num_samples_value} (Training)")
                    print(f"⏭️ Skipping validation due to training failure")
                    continue
                
                # Backup the trained model
                model_backup = backup_model_file(d_model_value, num_samples_value)
                
                # Run validation
                validation_success = run_validation()
                if not validation_success:
                    failed_runs.append(f"D_MODEL={d_model_value}_NUM_SAMPLES={num_samples_value} (Validation)")
                else:
                    successful_runs.append(f"D_MODEL={d_model_value}_NUM_SAMPLES={num_samples_value}")
                
                # Optional cleanup
                cleanup_model_file()
                
            except Exception as e:
                print(f"❌ Unexpected error for D_MODEL={d_model_value}, NUM_SAMPLES={num_samples_value}: {e}")
                failed_runs.append(f"D_MODEL={d_model_value}_NUM_SAMPLES={num_samples_value} (Exception)")
            
            run_duration = time.time() - run_start_time
            print(f"⏱️ Configuration {config_count} completed in {run_duration/60:.1f} minutes")
            
            # Brief pause between runs
            if config_count < total_configs:
                print("⏸️ Brief pause before next configuration...")
                time.sleep(2)
    
    # Restore original config
    restore_original_config()
    
    # Final summary
    total_duration = time.time() - total_start_time
    print(f"\n{'🎉'*30}")
    print("SWEEP COMPLETED!")
    print(f"{'🎉'*30}")
    print(f"⏱️ Total time: {total_duration/60:.1f} minutes")
    print(f"✅ Successful runs ({len(successful_runs)}):")
    for run in successful_runs:
        print(f"  ├─ {run}")
    
    if failed_runs:
        print(f"❌ Failed runs ({len(failed_runs)}):")
        for run in failed_runs:
            print(f"  ├─ {run}")
    else:
        print("🎊 All configurations completed successfully!")
    
    print(f"\n💡 Check your wandb project 'rwkv-regex-learning' for detailed results")
    print(f"📁 Model backups saved in: {current_dir}")
    
    # List backed up models
    model_backups = [f for f in os.listdir(current_dir) if f.startswith('rwkv7_fsm_experimental_model_d') and f.endswith('.pth')]
    if model_backups:
        print(f"📦 Model backups created:")
        for backup in sorted(model_backups):
            print(f"  ├─ {backup}")

if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n⏹️ Sweep interrupted by user")
        # Try to restore original config
        restore_original_config()
        sys.exit(1)
    except Exception as e:
        print(f"\n💥 Unexpected error: {e}")
        # Try to restore original config
        restore_original_config()

        sys.exit(1)