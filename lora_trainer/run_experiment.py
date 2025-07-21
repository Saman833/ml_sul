#!/usr/bin/env python3
"""
Simple wrapper to run LoRA bot detection experiments with comprehensive logging

Usage Examples:
    # Basic experiment
    python run_experiment.py
    
    # Custom experiment name
    python run_experiment.py --name "bot_detection_v2"
    
    # Different hyperparameters
    python run_experiment.py --name "high_rank_experiment" --lora_r 32 --lora_alpha 64
    
    # Quick test run
    python run_experiment.py --name "quick_test" --epochs 1 --batch_size 8
"""

import subprocess
import sys
from pathlib import Path

def run_experiment(experiment_name=None, **kwargs):
    """Run a LoRA experiment with the specified parameters"""
    
    # Build command
    cmd = [sys.executable, "experiment_tracker.py"]
    
    if experiment_name:
        cmd.extend(["--experiment_name", experiment_name])
    
    # Add other parameters
    for param, value in kwargs.items():
        if value is not None:
            cmd.extend([f"--{param}", str(value)])
    
    print("Running LoRA Bot Detection Experiment")
    print("=" * 60)
    print(f"Command: {' '.join(cmd)}")
    print("=" * 60)
    
    # Run the experiment
    try:
        result = subprocess.run(cmd, check=True)
        print("\nExperiment completed successfully!")
        return result.returncode
    except subprocess.CalledProcessError as e:
        print(f"\nExperiment failed with error: {e}")
        return e.returncode
    except KeyboardInterrupt:
        print("\nExiting...")
        return 1

def main():
    """Main function with predefined experiment configurations"""
    
    print("LoRA Bot Detection Experiment Runner")
    print("=" * 60)
    
    # Predefined experiment configurations
    experiments = {
        "baseline_experiment": {
            "description": "Baseline LoRA configuration",
            "params": {
                "lora_r": 16,
                "lora_alpha": 32,
                "learning_rate": 0.0003,
                "batch_size": 16,
                "epochs": 3
            }
        },
        "high_rank_experiment": {
            "description": "Higher LoRA rank for more capacity",
            "params": {
                "lora_r": 32,
                "lora_alpha": 64,
                "learning_rate": 0.0002,
                "batch_size": 16,
                "epochs": 3
            }
        },
        "low_rank_experiment": {
            "description": "Lower LoRA rank for efficiency",
            "params": {
                "lora_r": 8,
                "lora_alpha": 16,
                "learning_rate": 0.0005,
                "batch_size": 16,
                "epochs": 3
            }
        },
        "quick_test": {
            "description": "Quick test run (1 epoch)",
            "params": {
                "lora_r": 16,
                "lora_alpha": 32,
                "learning_rate": 0.0003,
                "batch_size": 8,
                "epochs": 1
            }
        }
    }

    print("Choose an experiment configuration:")
    for i, (name, config) in enumerate(experiments.items(), 1):
        params_str = ", ".join([f"{k}={v}" for k, v in config["params"].items()])
        print(f"{i}. {name}: {config['description']}")
        print(f"   Parameters: {params_str}")

    print("5. custom: Custom parameters (you'll be prompted)")

    try:
        choice = input("\nEnter your choice (1-5): ").strip()
        
        if choice in ["1", "2", "3", "4"]:
            # Use predefined experiment
            exp_names = list(experiments.keys())
            selected_name = exp_names[int(choice) - 1]
            config = experiments[selected_name]
            
            print(f"\nRunning experiment: {selected_name}")
            print(f"Description: {config['description']}")
            print(f"Parameters: {config['params']}")
            
            proceed = input("\nProceed? (y/N): ").strip().lower()
            if proceed == 'y':
                run_experiment(selected_name, **config["params"])
            else:
                print("Experiment cancelled.")
                
        elif choice == "5":
            # Custom experiment
            print("\nCustom experiment setup:")
            name = input("Experiment name: ").strip() or "custom_experiment"
            
            try:
                lora_r = int(input("LoRA rank (default: 16): ") or "16")
                lora_alpha = int(input("LoRA alpha (default: 32): ") or "32")
                learning_rate = float(input("Learning rate (default: 0.0003): ") or "0.0003")
                batch_size = int(input("Batch size (default: 16): ") or "16")
                epochs = int(input("Epochs (default: 3): ") or "3")
                
                print(f"\nRunning custom experiment: {name}")
                print(f"Parameters: lora_r={lora_r}, lora_alpha={lora_alpha}, learning_rate={learning_rate}, batch_size={batch_size}, epochs={epochs}")
                
                proceed = input("\nProceed? (y/N): ").strip().lower()
                if proceed == 'y':
                    run_experiment(name, lora_r=lora_r, lora_alpha=lora_alpha, 
                                 learning_rate=learning_rate, batch_size=batch_size, epochs=epochs)
                else:
                    print("Experiment cancelled.")
                    
            except ValueError:
                print("Invalid input. Using defaults.")
                run_experiment("custom_experiment")
        else:
            print("Invalid choice. Exiting.")
            
    except KeyboardInterrupt:
        print("\nExiting...")

if __name__ == "__main__":
    main() 