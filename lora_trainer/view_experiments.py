#!/usr/bin/env python3
"""
MLflow Experiment Viewer for LoRA Bot Detection

This script helps you view and manage your MLflow experiments.

Usage:
    python view_experiments.py
"""

import os
import subprocess
import sys
from pathlib import Path
import mlflow
import pandas as pd

def find_experiment_dirs():
    """Find all experiment directories with MLflow data"""
    experiments_dir = Path("./experiments")
    experiment_dirs = []
    
    if experiments_dir.exists():
        for exp_dir in experiments_dir.iterdir():
            if exp_dir.is_dir() and (exp_dir / "mlruns").exists():
                experiment_dirs.append(exp_dir)
    
    return experiment_dirs

def list_experiments():
    """List all available experiments with their metrics"""
    experiment_dirs = find_experiment_dirs()
    
    if not experiment_dirs:
        print("No MLflow experiments found in ./experiments/")
        return []
    
    print("Available Experiments:")
    print("=" * 80)
    
    for i, exp_dir in enumerate(experiment_dirs, 1):
        try:
            # Set MLflow tracking URI
            tracking_uri = (exp_dir / "mlruns").absolute().as_uri()
            mlflow.set_tracking_uri(tracking_uri)
            
            # Get experiment info
            experiment = mlflow.get_experiment_by_name(exp_dir.name)
            if experiment:
                runs = mlflow.search_runs(experiment_ids=[experiment.experiment_id])
                if not runs.empty:
                    latest_run = runs.iloc[0]
                    
                    print(f"{i}. {exp_dir.name}")
                    print(f"   Status: {latest_run.get('status', 'Unknown')}")
                    print(f"   Start Time: {latest_run.get('start_time', 'N/A')}")
                    print(f"   Duration: {latest_run.get('end_time', 'N/A')}")
                    print(f"   Latest Accuracy: {latest_run.get('metrics.final_eval_accuracy', 'N/A')}")
                    print()
                else:
                    print(f"{i}. {exp_dir.name} (No runs found)")
            else:
                print(f"{i}. {exp_dir.name} (Error reading: No experiment found)")
                
        except Exception as e:
            print(f"{i}. {exp_dir.name} (Error reading: {e})")
    
    return experiment_dirs

def start_mlflow_ui(experiment_dir):
    """Start MLflow UI for a specific experiment"""
    mlflow_dir = experiment_dir / "mlruns"
    
    if not mlflow_dir.exists():
        print(f"No MLflow data found in {experiment_dir}")
        return
    
    print(f"Starting MLflow UI for: {experiment_dir.name}")
    print(f"Tracking URI: file://{mlflow_dir.absolute()}")
    print("Opening browser at: http://localhost:5000")
    print("\nPress Ctrl+C to stop the MLflow UI")
    print("=" * 60)
    
    try:
        # Use proper URI format for Windows
        tracking_uri = mlflow_dir.absolute().as_uri()
        cmd = ["mlflow", "ui", "--backend-store-uri", tracking_uri]
        subprocess.run(cmd)
        
    except KeyboardInterrupt:
        print("\nMLflow UI stopped.")
    except Exception as e:
        print(f"Error starting MLflow UI: {e}")

def compare_experiments():
    """Compare metrics across experiments"""
    experiment_dirs = find_experiment_dirs()
    
    if not experiment_dirs:
        print("No experiments found")
        return
    
    print("Experiment Comparison:")
    print("=" * 80)
    
    comparison_data = []
    
    for exp_dir in experiment_dirs:
        try:
            # Set MLflow tracking URI
            tracking_uri = (exp_dir / "mlruns").absolute().as_uri()
            mlflow.set_tracking_uri(tracking_uri)
            
            # Get experiment info
            experiment = mlflow.get_experiment_by_name(exp_dir.name)
            if experiment:
                runs = mlflow.search_runs(experiment_ids=[experiment.experiment_id])
                if not runs.empty:
                    latest_run = runs.iloc[0]
                    
                    comparison_data.append({
                        "Experiment": exp_dir.name,
                        "Accuracy": latest_run.get('metrics.accuracy', 0),
                        "F1 Score": latest_run.get('metrics.f1_score', 0),
                        "Loss": latest_run.get('metrics.loss', 0),
                        "Learning Rate": latest_run.get('params.learning_rate', 'N/A'),
                        "LoRA Rank": latest_run.get('params.lora_r', 'N/A'),
                        "Batch Size": latest_run.get('params.batch_size', 'N/A'),
                        "Status": latest_run.get('status', 'Unknown')
                    })
                    
        except Exception as e:
            print(f"Error reading {exp_dir.name}: {e}")
    
    if comparison_data:
        df = pd.DataFrame(comparison_data)
        df = df.sort_values('Accuracy', ascending=False)
        
        print(df.to_string(index=False))
        print(f"\nBest performing experiment: {df.iloc[0]['Experiment']}")
        print(f"   Accuracy: {df.iloc[0]['Accuracy']:.4f}")
        print(f"   F1 Score: {df.iloc[0]['F1 Score']:.4f}")
        print(f"   Loss: {df.iloc[0]['Loss']:.4f}")
    else:
        print("No experiment data found")

def main():
    """Main function"""
    print("MLflow Experiment Viewer")
    print("=" * 40)
    
    while True:
        print("\nOptions:")
        print("1. List all experiments")
        print("2. Start MLflow UI for an experiment")
        print("3. Compare experiments")
        print("4. Exit")
        
        try:
            choice = input("\nEnter choice (1-4): ").strip()
            
            if choice == "1":
                list_experiments()
                
            elif choice == "2":
                experiment_dirs = list_experiments()
                if experiment_dirs:
                    try:
                        selection = int(input(f"\nSelect experiment (1-{len(experiment_dirs)}): ")) - 1
                        if 0 <= selection < len(experiment_dirs):
                            start_mlflow_ui(experiment_dirs[selection])
                        else:
                            print("Invalid selection")
                    except ValueError:
                        print("Please enter a valid number")
                        
            elif choice == "3":
                compare_experiments()
                
            elif choice == "4":
                break
                
            else:
                print("Invalid choice. Please enter 1-4.")
                
        except KeyboardInterrupt:
            print("\nExiting...")
            break

if __name__ == "__main__":
    main() 