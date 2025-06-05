"""
Pipeline Configuration and Helper Functions for RWKV Regex Learning Project

This module provides utilities to:
1. Manage dataset artifacts across pipeline stages
2. Link training runs to datasets
3. Link validation runs to training runs
4. Provide easy pipeline configuration
"""

import json
import os
from typing import Any, Dict, Optional, Tuple

import wandb


class RWKVPipelineManager:
    """Manages the pipeline flow between dataset generation, training, and validation"""
    def __init__(self, project_name: str = "rwkv-regex-learning"):
        self.project_name = project_name
        self.api = wandb.Api()
        
    def find_latest_dataset_artifact(self, filters: Optional[Dict] = None) -> Optional[str]:
        """Find the latest dataset artifact matching optional filters"""
        try:
            # Get all artifacts of type dataset from the project
            project = self.api.project(self.project_name)
            artifacts = list(project.artifacts(type_name="dataset"))
            
            if filters:
                # Filter artifacts based on metadata
                filtered_artifacts = []
                for artifact in artifacts:
                    if all(artifact.metadata.get(k) == v for k, v in filters.items()):
                        filtered_artifacts.append(artifact)
                artifacts = filtered_artifacts
            
            if artifacts:
                # Return the most recent artifact
                latest = sorted(artifacts, key=lambda x: x.created_at, reverse=True)[0]
                return f"{latest.name}:latest"
            return None
        except Exception as e:
            print(f"Error finding dataset artifact: {e}")
            return None
    def find_latest_model_artifact(self, filters: Optional[Dict] = None) -> Optional[str]:
        """Find the latest model artifact matching optional filters"""
        try:
            project = self.api.project(self.project_name)
            artifacts = list(project.artifacts(type_name="model"))
            
            if filters:
                # Filter artifacts based on metadata
                filtered_artifacts = []
                for artifact in artifacts:
                    if all(artifact.metadata.get(k) == v for k, v in filters.items()):
                        filtered_artifacts.append(artifact)
                artifacts = filtered_artifacts
            
            if artifacts:
                # Return the most recent artifact
                latest = sorted(artifacts, key=lambda x: x.created_at, reverse=True)[0]
                return f"{latest.name}:latest"
            return None
        except Exception as e:
            print(f"Error finding model artifact: {e}")
            return None
    
    def find_latest_training_run(self, filters: Optional[Dict] = None) -> Optional[Dict]:
        """Find the latest training run matching optional filters"""
        try:
            runs = self.api.runs(
                path=self.project_name,
                filters={"config.job_type": "training"}
            )
            
            if filters:
                # Filter runs based on config
                filtered_runs = []
                for run in runs:
                    if all(run.config.get(k) == v for k, v in filters.items()):
                        filtered_runs.append(run)
                runs = filtered_runs
            
            if runs:
                latest_run = runs[0]  # Most recent
                return {
                    "id": latest_run.id,
                    "name": latest_run.name,
                    "config": dict(latest_run.config),
                    "summary": dict(latest_run.summary)
                }
            return None
        except Exception as e:
            print(f"Error finding training run: {e}")
            return None
    
    def get_dataset_config_from_artifact(self, artifact_name: str) -> Optional[Dict]:
        """Get dataset configuration from artifact metadata"""
        try:
            artifact = self.api.artifact(f"{self.project_name}/{artifact_name}")
            return artifact.metadata
        except Exception as e:
            print(f"Error getting dataset config: {e}")
            return None
    
    def create_pipeline_config(self, 
                             dataset_samples: Optional[int] = None,
                             dataset_max_len: Optional[int] = None,
                             use_existing_dataset: bool = True,
                             training_hyperparams: Optional[Dict] = None,
                             use_existing_model: bool = True) -> Dict[str, Any]:
        """
        Create a pipeline configuration that determines what to reuse vs regenerate
        
        Args:
            dataset_samples: If specified, use/create dataset with this sample count
            dataset_max_len: If specified, use/create dataset with this max length
            use_existing_dataset: Whether to try to reuse existing dataset
            training_hyperparams: If specified, use these hyperparameters for training
            use_existing_model: Whether to try to reuse existing model
        
        Returns:
            Dictionary with pipeline configuration
        """
        config = {
            "dataset": {
                "use_existing": use_existing_dataset,
                "artifact_name": None,
                "filters": {}
            },
            "training": {
                "use_existing": use_existing_model,
                "hyperparams": training_hyperparams or {},
                "artifact_name": None
            },
            "validation": {
                "link_to_training": True
            }
        }
        
        # Set dataset filters if specified
        if dataset_samples is not None:
            config["dataset"]["filters"]["num_samples"] = dataset_samples
        if dataset_max_len is not None:
            config["dataset"]["filters"]["max_len"] = dataset_max_len
        
        # Find existing dataset if requested
        if use_existing_dataset:
            dataset_artifact = self.find_latest_dataset_artifact(config["dataset"]["filters"])
            if dataset_artifact:
                config["dataset"]["artifact_name"] = dataset_artifact
                print(f"✓ Found existing dataset: {dataset_artifact}")
            else:
                print("⚠ No matching existing dataset found, will create new one")
                config["dataset"]["use_existing"] = False
        
        # Find existing model if requested
        if use_existing_model and training_hyperparams:
            # Create filters based on hyperparameters
            model_filters = {}
            for key, value in training_hyperparams.items():
                if key.upper() in ["D_MODEL", "N_LAYER", "LEARNING_RATE", "BATCH_SIZE"]:
                    model_filters[key.lower()] = value
            
            model_artifact = self.find_latest_model_artifact(model_filters)
            if model_artifact:
                config["training"]["artifact_name"] = model_artifact
                print(f"✓ Found existing model: {model_artifact}")
            else:
                print("⚠ No matching existing model found, will train new one")
                config["training"]["use_existing"] = False
        
        return config

# Global pipeline manager instance
pipeline_manager = RWKVPipelineManager()

def setup_dataset_generation(num_samples: int, max_len: int, target_substring: str = "abbccc") -> Dict[str, Any]:
    """Setup configuration for dataset generation stage"""
    # Check if we already have this exact dataset
    existing_artifact = pipeline_manager.find_latest_dataset_artifact({
        "num_samples": num_samples,
        "max_len": max_len,
        "target_substring": target_substring
    })
    
    if existing_artifact:
        print(f"✓ Found existing dataset with same parameters: {existing_artifact}")
        print("  You can skip dataset generation and use this artifact in training.")
        return {
            "should_generate": False,
            "existing_artifact": existing_artifact,
            "config": {
                "num_samples": num_samples,
                "max_len": max_len,
                "target_substring": target_substring
            }
        }
    
    print(f"⚠ No existing dataset found with samples={num_samples}, max_len={max_len}")
    print("  Will generate new dataset.")
    return {
        "should_generate": True,
        "existing_artifact": None,
        "config": {
            "num_samples": num_samples,
            "max_len": max_len,
            "target_substring": target_substring
        }
    }

def setup_training(dataset_artifact_name: Optional[str] = None, 
                  hyperparams: Optional[Dict] = None,
                  reuse_existing_model: bool = True) -> Dict[str, Any]:
    """Setup configuration for training stage"""
    config = {
        "dataset_artifact": dataset_artifact_name,
        "use_wandb_dataset": dataset_artifact_name is not None,
        "hyperparams": hyperparams or {},
        "existing_model_artifact": None
    }
    
    # If no dataset artifact specified, try to find the latest one
    if dataset_artifact_name is None:
        latest_dataset = pipeline_manager.find_latest_dataset_artifact()
        if latest_dataset:
            config["dataset_artifact"] = latest_dataset
            config["use_wandb_dataset"] = True
            print(f"✓ Will use latest dataset: {latest_dataset}")
        else:
            print("⚠ No dataset artifacts found, will use local file")
    
    # Check for existing model with same hyperparameters
    if reuse_existing_model and hyperparams:
        existing_model = pipeline_manager.find_latest_model_artifact(hyperparams)
        if existing_model:
            config["existing_model_artifact"] = existing_model
            print(f"✓ Found existing model with same hyperparameters: {existing_model}")
            print("  You can skip training and use this model for validation.")
    
    return config

def setup_validation(model_artifact_name: Optional[str] = None,
                    training_run_id: Optional[str] = None) -> Dict[str, Any]:
    """Setup configuration for validation stage"""
    config = {
        "model_artifact": model_artifact_name,
        "use_wandb_model": model_artifact_name is not None,
        "training_run_id": training_run_id
    }
    
    # If no model artifact specified, try to find the latest one
    if model_artifact_name is None:
        latest_model = pipeline_manager.find_latest_model_artifact()
        if latest_model:
            config["model_artifact"] = latest_model
            config["use_wandb_model"] = True
            print(f"✓ Will use latest model: {latest_model}")
        else:
            print("⚠ No model artifacts found, will use local file")
    
    # If no training run specified, try to find the latest one
    if training_run_id is None:
        latest_training = pipeline_manager.find_latest_training_run()
        if latest_training:
            config["training_run_id"] = latest_training["id"]
            print(f"✓ Will link to latest training run: {latest_training['name']}")
    
    return config

# Example usage configurations
EXAMPLE_CONFIGS = {
    "full_pipeline_new": {
        "description": "Generate new dataset, train new model, validate",
        "dataset": {"num_samples": 1000, "max_len": 50, "generate_new": True},
        "training": {"use_new_dataset": True, "hyperparams": {"learning_rate": 1e-3}},
        "validation": {"use_new_model": True}
    },
    
    "reuse_dataset_new_training": {
        "description": "Reuse existing dataset, train with different hyperparameters",
        "dataset": {"use_existing": True},
        "training": {"hyperparams": {"learning_rate": 5e-4, "batch_size": 512}},
        "validation": {"use_new_model": True}
    },
    
    "reuse_everything": {
        "description": "Reuse existing dataset and model, just run validation",
        "dataset": {"use_existing": True},
        "training": {"skip": True, "use_existing_model": True},
        "validation": {"use_existing_model": True}
    }
}

if __name__ == "__main__":
    print("RWKV Pipeline Manager")
    print("=" * 50)
    
    # Example: Check what datasets are available
    manager = RWKVPipelineManager()
    print("\nAvailable datasets:")
    try:
        project = manager.api.project("rwkv-regex-learning")
        artifacts = list(project.artifacts(type_name="dataset"))
        for artifact in artifacts[:5]:  # Show first 5
            print(f"  {artifact.name} - {artifact.metadata}")
    except Exception:
        print("  No datasets found or error accessing wandb")
    
    print("\nAvailable models:")
    try:
        project = manager.api.project("rwkv-regex-learning")
        artifacts = list(project.artifacts(type_name="model"))
        for artifact in artifacts[:5]:  # Show first 5
            print(f"  {artifact.name} - {artifact.metadata}")
    except Exception:
        print("  No models found or error accessing wandb")
    
    print("\nExample pipeline configurations:")
    for name, config in EXAMPLE_CONFIGS.items():
        print(f"  {name}: {config['description']}")