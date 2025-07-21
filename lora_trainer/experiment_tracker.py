

import json
import os
import time
import argparse
from datetime import datetime
from pathlib import Path
import pandas as pd
import numpy as np
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import torch
from transformers import (
    AutoTokenizer, 
    AutoModelForSequenceClassification,
    TrainingArguments, 
    Trainer,
    EarlyStoppingCallback
)
from peft import LoraConfig, get_peft_model, TaskType, PeftModel
from datasets import Dataset, DatasetDict
import mlflow
import mlflow.pytorch
import mlflow.transformers
import warnings
warnings.filterwarnings('ignore')

# Disable transformers automatic MLflow integration to prevent conflicts
import os
os.environ["DISABLE_MLFLOW_INTEGRATION"] = "TRUE"

class LoRAExperimentTracker:
    """Comprehensive experiment tracking for LoRA training with MLflow"""
    
    def __init__(self, experiment_name=None, output_dir="./experiments"):
        """Initialize experiment tracker with MLflow"""
        self.experiment_name = experiment_name or f"lora_bot_detection_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        self.output_dir = Path(output_dir) / self.experiment_name
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # Set MLflow tracking URI to local directory (Windows-compatible)
        mlflow_dir = self.output_dir / "mlruns"
        mlflow_dir.mkdir(exist_ok=True)
        
        # Use proper URI format for Windows
        import platform
        if platform.system() == "Windows":
            # Convert Windows path to proper file URI
            mlflow_uri = mlflow_dir.absolute().as_uri()
            mlflow.set_tracking_uri(mlflow_uri)
        else:
            mlflow.set_tracking_uri(f"file://{mlflow_dir.absolute()}")
        
        # Set or create MLflow experiment
        try:
            experiment = mlflow.get_experiment_by_name(self.experiment_name)
            if experiment is None:
                experiment_id = mlflow.create_experiment(self.experiment_name)
            else:
                experiment_id = experiment.experiment_id
        except Exception:
            experiment_id = mlflow.create_experiment(self.experiment_name)
        
        mlflow.set_experiment(self.experiment_name)
        
        # Initialize experiment metadata (for backup)
        self.metadata = {
            "experiment_info": {
                "name": self.experiment_name,
                "timestamp": datetime.now().isoformat(),
                "script_version": "1.0.0",
                "mlflow_experiment_id": experiment_id
            },
            "data_info": {},
            "model_info": {},
            "training_info": {},
            "results": {},
            "artifacts": {}
        }
        
        print(f"Experiment: {self.experiment_name}")
        
        # Start MLflow run
        self.run = mlflow.start_run(run_name=f"{self.experiment_name}_{datetime.now().strftime('%H%M%S')}")
        print(f"Run ID: {self.run.info.run_id}")
    
    def log_data_info(self, train_texts, train_labels, val_texts, val_labels):
        """Log dataset information to MLflow and metadata"""
        train_size = len(train_texts)
        val_size = len(val_texts)
        
        # Calculate class distribution
        train_label_counts = pd.Series(train_labels).value_counts().to_dict()
        val_label_counts = pd.Series(val_labels).value_counts().to_dict()
        
        # Log to MLflow
        mlflow.log_param("training_samples", train_size)
        mlflow.log_param("validation_samples", val_size)
        mlflow.log_param("total_samples", train_size + val_size)
        mlflow.log_param("train_split_ratio", train_size / (train_size + val_size))
        mlflow.log_param("val_split_ratio", val_size / (train_size + val_size))
        mlflow.log_param("train_human_samples", train_label_counts.get(0, 0))
        mlflow.log_param("train_bot_samples", train_label_counts.get(1, 0))
        mlflow.log_param("val_human_samples", val_label_counts.get(0, 0))
        mlflow.log_param("val_bot_samples", val_label_counts.get(1, 0))
        
        # Store in metadata (backup)
        self.metadata["data_info"] = {
            "training_samples": train_size,
            "validation_samples": val_size,
            "total_samples": train_size + val_size,
            "train_class_distribution": train_label_counts,
            "val_class_distribution": val_label_counts,
            "class_names": {0: "human", 1: "bot"},
            "train_split_ratio": train_size / (train_size + val_size),
            "val_split_ratio": val_size / (train_size + val_size)
        }
        
        print(f"📊 Dataset Info Logged to MLflow:")
        print(f"   Training samples: {train_size}")
        print(f"   Validation samples: {val_size}")
        print(f"   Train class distribution: {train_label_counts}")
        print(f"   Val class distribution: {val_label_counts}")
    
    def log_model_info(self, base_model_name, tokenizer_name, lora_config):
        """Log model and LoRA configuration to MLflow"""
        # Log to MLflow
        mlflow.log_param("base_model_name", base_model_name)
        mlflow.log_param("tokenizer_name", tokenizer_name)
        mlflow.log_param("lora_rank", lora_config.r)
        mlflow.log_param("lora_alpha", lora_config.lora_alpha)
        mlflow.log_param("lora_dropout", lora_config.lora_dropout)
        mlflow.log_param("lora_target_modules", str(lora_config.target_modules))
        mlflow.log_param("lora_task_type", str(lora_config.task_type))
        mlflow.log_param("lora_bias", lora_config.bias)
        mlflow.log_param("lora_fan_in_fan_out", lora_config.fan_in_fan_out)
        
        # Store in metadata (backup)
        self.metadata["model_info"] = {
            "base_model_name": base_model_name,
            "tokenizer_name": tokenizer_name,
            "lora_config": {
                "r": lora_config.r,
                "lora_alpha": lora_config.lora_alpha,
                "lora_dropout": lora_config.lora_dropout,
                "target_modules": lora_config.target_modules,
                "task_type": str(lora_config.task_type),
                "bias": lora_config.bias,
                "fan_in_fan_out": lora_config.fan_in_fan_out
            }
        }
        
        print(f"🤖 Model Info Logged to MLflow:")
        print(f"   Base model: {base_model_name}")
        print(f"   LoRA rank (r): {lora_config.r}")
        print(f"   LoRA alpha: {lora_config.lora_alpha}")
        print(f"   LoRA dropout: {lora_config.lora_dropout}")
        print(f"   Target modules: {lora_config.target_modules}")
    
    def log_training_info(self, training_args):
        """Log training configuration to MLflow"""
        # Log to MLflow
        mlflow.log_param("learning_rate", training_args.learning_rate)
        mlflow.log_param("batch_size", training_args.per_device_train_batch_size)
        mlflow.log_param("num_epochs", training_args.num_train_epochs)
        mlflow.log_param("warmup_steps", training_args.warmup_steps)
        mlflow.log_param("weight_decay", training_args.weight_decay)
        mlflow.log_param("max_length", 128)
        mlflow.log_param("optimizer", "AdamW")
        mlflow.log_param("lr_scheduler", "linear")
        mlflow.log_param("gradient_accumulation_steps", training_args.gradient_accumulation_steps)
        mlflow.log_param("fp16", training_args.fp16)
        mlflow.log_param("evaluation_strategy", training_args.evaluation_strategy)
        mlflow.log_param("save_strategy", training_args.save_strategy)
        mlflow.log_param("logging_steps", training_args.logging_steps)
        
        # Store in metadata (backup)
        self.metadata["training_info"] = {
            "learning_rate": training_args.learning_rate,
            "batch_size": training_args.per_device_train_batch_size,
            "num_epochs": training_args.num_train_epochs,
            "warmup_steps": training_args.warmup_steps,
            "weight_decay": training_args.weight_decay,
            "max_length": 128,  # From tokenization
            "optimizer": "AdamW",  # Default for Trainer
            "lr_scheduler": "linear",  # Default
            "gradient_accumulation_steps": training_args.gradient_accumulation_steps,
            "fp16": training_args.fp16,
            "evaluation_strategy": training_args.evaluation_strategy,
            "save_strategy": training_args.save_strategy,
            "logging_steps": training_args.logging_steps
        }
        
        print(f"🎛️ Training Config Logged to MLflow:")
        print(f"   Learning rate: {training_args.learning_rate}")
        print(f"   Batch size: {training_args.per_device_train_batch_size}")
        print(f"   Epochs: {training_args.num_train_epochs}")
    
    def log_training_results(self, trainer, eval_results):
        """Log training and validation results to MLflow"""
        # Get training history
        log_history = trainer.state.log_history
        
        # Extract metrics
        train_losses = [log['train_loss'] for log in log_history if 'train_loss' in log]
        eval_losses = [log['eval_loss'] for log in log_history if 'eval_loss' in log]
        eval_accuracies = [log['eval_accuracy'] for log in log_history if 'eval_accuracy' in log]
        
        # Log final metrics to MLflow
        mlflow.log_metric("final_train_loss", train_losses[-1] if train_losses else 0)
        mlflow.log_metric("final_eval_loss", eval_results.get('eval_loss', 0))
        mlflow.log_metric("final_eval_accuracy", eval_results.get('eval_accuracy', 0))
        mlflow.log_metric("final_eval_f1", eval_results.get('eval_f1', 0))
        mlflow.log_metric("final_eval_precision", eval_results.get('eval_precision', 0))
        mlflow.log_metric("final_eval_recall", eval_results.get('eval_recall', 0))
        mlflow.log_metric("training_time_seconds", trainer.state.log_history[-1].get('train_runtime', 0))
        mlflow.log_metric("total_training_steps", trainer.state.max_steps)
        
        # Log training history (step by step)
        for i, loss in enumerate(train_losses):
            mlflow.log_metric("train_loss", loss, step=i)
        for i, loss in enumerate(eval_losses):
            mlflow.log_metric("eval_loss", loss, step=i)
        for i, acc in enumerate(eval_accuracies):
            mlflow.log_metric("eval_accuracy", acc, step=i)
        
        # Store in metadata (backup)
        self.metadata["results"] = {
            "final_train_loss": train_losses[-1] if train_losses else None,
            "final_eval_loss": eval_results.get('eval_loss'),
            "final_eval_accuracy": eval_results.get('eval_accuracy'),
            "final_eval_f1": eval_results.get('eval_f1'),
            "final_eval_precision": eval_results.get('eval_precision'),
            "final_eval_recall": eval_results.get('eval_recall'),
            "training_time_seconds": trainer.state.log_history[-1].get('train_runtime', 0),
            "total_training_steps": trainer.state.max_steps,
            "best_model_checkpoint": str(trainer.state.best_model_checkpoint) if trainer.state.best_model_checkpoint else None,
            "training_history": {
                "train_losses": train_losses,
                "eval_losses": eval_losses,
                "eval_accuracies": eval_accuracies
            }
        }
        
        print(f"📈 Training Results Logged to MLflow:")
        print(f"   Final eval accuracy: {eval_results.get('eval_accuracy', 'N/A'):.4f}")
        print(f"   Final eval loss: {eval_results.get('eval_loss', 'N/A'):.4f}")
        print(f"   Final eval F1: {eval_results.get('eval_f1', 'N/A'):.4f}")
    
    def log_model_artifacts(self, model_path, lora_weights_info, model=None):
        """Log model artifacts to MLflow"""
        model_size_mb = self._get_directory_size(model_path)
        
        # Log model metadata to MLflow
        mlflow.log_param("model_save_path", str(model_path))
        mlflow.log_param("model_size_mb", model_size_mb)
        mlflow.log_param("trainable_params", lora_weights_info["trainable_params"])
        mlflow.log_param("total_params", lora_weights_info["total_params"])
        mlflow.log_param("trainable_percentage", lora_weights_info["trainable_percentage"])
        
        # Log model artifacts as MLflow artifacts
        try:
            mlflow.log_artifacts(str(model_path), "model")
            print(f"📦 Model artifacts uploaded to MLflow")
        except Exception as e:
            print(f"⚠️ Could not upload model artifacts to MLflow: {e}")
        
        # Log the model itself if provided
        if model is not None:
            try:
                mlflow.pytorch.log_model(
                    model,
                    "lora_model",
                    registered_model_name=f"lora_bot_detector_{datetime.now().strftime('%Y%m%d')}"
                )
                print(f"🤖 LoRA model logged to MLflow Model Registry")
            except Exception as e:
                print(f"⚠️ Could not log model to MLflow: {e}")
        
        # Store in metadata (backup)
        self.metadata["artifacts"] = {
            "model_save_path": str(model_path),
            "lora_weights_file": "adapter_model.safetensors",
            "lora_config_file": "adapter_config.json",
            "tokenizer_files": ["tokenizer.json", "vocab.txt", "tokenizer_config.json"],
            "lora_weights_info": lora_weights_info,
            "model_size_mb": model_size_mb,
            "saved_at": datetime.now().isoformat()
        }
        
        print(f"💾 Model Artifacts Logged to MLflow:")
        print(f"   Model path: {model_path}")
        print(f"   Model size: {model_size_mb:.2f} MB")
    
    def _get_directory_size(self, path):
        """Calculate directory size in MB"""
        total_size = 0
        for dirpath, dirnames, filenames in os.walk(path):
            for filename in filenames:
                filepath = os.path.join(dirpath, filename)
                total_size += os.path.getsize(filepath)
        return total_size / (1024 * 1024)  # Convert to MB
    
    def save_experiment_metadata(self):
        """Save all experiment metadata to JSON file"""
        metadata_file = self.output_dir / "experiment_metadata.json"
        
        # Convert sets to lists for JSON serialization
        def convert_sets_to_lists(obj):
            if isinstance(obj, dict):
                return {k: convert_sets_to_lists(v) for k, v in obj.items()}
            elif isinstance(obj, list):
                return [convert_sets_to_lists(item) for item in obj]
            elif isinstance(obj, set):
                return list(obj)
            else:
                return obj
        
        serializable_metadata = convert_sets_to_lists(self.metadata)
        
        with open(metadata_file, 'w', encoding='utf-8') as f:
            json.dump(serializable_metadata, f, indent=2, ensure_ascii=False)
        
        print(f"💾 Experiment metadata saved: {metadata_file}")
        return metadata_file
    
    def generate_experiment_report(self):
        """Generate a comprehensive experiment report"""
        report_file = self.output_dir / "experiment_report.md"
        
        report_content = f"""# LoRA Bot Detection Experiment Report

## Experiment Information
- **Name**: {self.metadata['experiment_info']['name']}
- **Timestamp**: {self.metadata['experiment_info']['timestamp']}
- **Script Version**: {self.metadata['experiment_info']['script_version']}

## Dataset Information
- **Training Samples**: {self.metadata['data_info']['training_samples']:,}
- **Validation Samples**: {self.metadata['data_info']['validation_samples']:,}
- **Total Samples**: {self.metadata['data_info']['total_samples']:,}
- **Train/Val Split**: {self.metadata['data_info']['train_split_ratio']:.2%} / {self.metadata['data_info']['val_split_ratio']:.2%}

### Class Distribution
**Training Set**:
- Human: {self.metadata['data_info']['train_class_distribution'].get(0, 0):,}
- Bot: {self.metadata['data_info']['train_class_distribution'].get(1, 0):,}

**Validation Set**:
- Human: {self.metadata['data_info']['val_class_distribution'].get(0, 0):,}
- Bot: {self.metadata['data_info']['val_class_distribution'].get(1, 0):,}

## Model Configuration
- **Base Model**: {self.metadata['model_info']['base_model_name']}
- **Tokenizer**: {self.metadata['model_info']['tokenizer_name']}

### LoRA Hyperparameters
- **Rank (r)**: {self.metadata['model_info']['lora_config']['r']}
- **Alpha**: {self.metadata['model_info']['lora_config']['lora_alpha']}
- **Dropout**: {self.metadata['model_info']['lora_config']['lora_dropout']}
- **Target Modules**: {', '.join(self.metadata['model_info']['lora_config']['target_modules'])}
- **Bias**: {self.metadata['model_info']['lora_config']['bias']}

## Training Configuration
- **Learning Rate**: {self.metadata['training_info']['learning_rate']}
- **Batch Size**: {self.metadata['training_info']['batch_size']}
- **Epochs**: {self.metadata['training_info']['num_epochs']}
- **Warmup Steps**: {self.metadata['training_info']['warmup_steps']}
- **Weight Decay**: {self.metadata['training_info']['weight_decay']}
- **Max Sequence Length**: {self.metadata['training_info']['max_length']}

## Results
- **Final Validation Accuracy**: {self.metadata['results']['final_eval_accuracy']:.4f}
- **Final Validation Loss**: {self.metadata['results']['final_eval_loss']:.4f}
- **Final Validation F1**: {self.metadata['results']['final_eval_f1']:.4f}
- **Final Validation Precision**: {self.metadata['results']['final_eval_precision']:.4f}
- **Final Validation Recall**: {self.metadata['results']['final_eval_recall']:.4f}
- **Training Time**: {self.metadata['results']['training_time_seconds']:.2f} seconds
- **Total Training Steps**: {self.metadata['results']['total_training_steps']}

## Model Artifacts
- **Save Path**: {self.metadata['artifacts']['model_save_path']}
- **Model Size**: {self.metadata['artifacts']['model_size_mb']:.2f} MB
- **LoRA Weights File**: {self.metadata['artifacts']['lora_weights_file']}
- **Config File**: {self.metadata['artifacts']['lora_config_file']}

## Usage Instructions
```bash
# Load the trained model
python test_bot_detector.py

# Or use in your code:
from peft import PeftModel, PeftConfig
config = PeftConfig.from_pretrained("{self.metadata['artifacts']['model_save_path']}")
model = PeftModel.from_pretrained(base_model, "{self.metadata['artifacts']['model_save_path']}")
```

---
*Report generated on {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}*
"""
        
        with open(report_file, 'w', encoding='utf-8') as f:
            f.write(report_content)
        
        print(f"📄 Experiment report generated: {report_file}")
        return report_file
    
    def end_run(self):
        """End MLflow run and cleanup"""
        try:
            mlflow.end_run()
            print(f"✅ MLflow run ended successfully")
        except Exception as e:
            print(f"⚠️ Error ending MLflow run: {e}")
    
    def get_mlflow_ui_command(self):
        """Get command to start MLflow UI (Windows-compatible)"""
        mlflow_dir = self.output_dir / "mlruns"
        
        # Use proper URI format for Windows
        import platform
        if platform.system() == "Windows":
            mlflow_uri = mlflow_dir.absolute().as_uri()
            return f"mlflow ui --backend-store-uri \"{mlflow_uri}\""
        else:
            return f"mlflow ui --backend-store-uri file://{mlflow_dir.absolute()}"

def compute_metrics(eval_pred):
    """Compute accuracy, precision, recall, F1 for evaluation"""
    predictions, labels = eval_pred
    predictions = np.argmax(predictions, axis=1)
    
    accuracy = accuracy_score(labels, predictions)
    report = classification_report(labels, predictions, output_dict=True, zero_division=0)
    
    return {
        'accuracy': accuracy,
        'f1': report['macro avg']['f1-score'],
        'precision': report['macro avg']['precision'],
        'recall': report['macro avg']['recall']
    }

def main():
    # Disable automatic MLflow integration in transformers to avoid conflicts
    import os
    os.environ["DISABLE_MLFLOW_INTEGRATION"] = "TRUE"
    
    parser = argparse.ArgumentParser(description='LoRA Bot Detection Experiment Tracker')
    parser.add_argument('--experiment_name', type=str, help='Name for this experiment')
    parser.add_argument('--base_model', type=str, default='distilbert-base-uncased', help='Base model name')
    parser.add_argument('--lora_r', type=int, default=16, help='LoRA rank')
    parser.add_argument('--lora_alpha', type=int, default=32, help='LoRA alpha')
    parser.add_argument('--learning_rate', type=float, default=3e-4, help='Learning rate')
    parser.add_argument('--batch_size', type=int, default=16, help='Batch size')
    parser.add_argument('--epochs', type=int, default=3, help='Number of epochs')
    args = parser.parse_args()
    
    # Initialize experiment tracker
    tracker = LoRAExperimentTracker(args.experiment_name)
    
    print("🚀 Starting LoRA Bot Detection Training with Comprehensive Logging")
    print("=" * 80)
    
    # Load and prepare data
    print("📂 Loading data...")
    
    # Load the JSON file with conversations
    with open("train.json", 'r', encoding='utf-8') as f:
        conversations = json.load(f)
    
    # Load the CSV file with labels
    labels_df = pd.read_csv("ytrain.csv")
    
    # Create the training dataset by combining text and labels
    texts = []
    labels = []
    
    for _, row in labels_df.iterrows():
        dialog_id = row['dialog_id']
        participant_index = str(row['participant_index'])
        is_bot = row['is_bot']
        
        # Find the corresponding conversation
        if dialog_id in conversations:
            dialog = conversations[dialog_id]
            # Get all messages from this participant
            participant_texts = []
            for message in dialog:
                if message['participant_index'] == participant_index:
                    participant_texts.append(message['text'])
            
            # Combine all messages from this participant into one text
            if participant_texts:
                combined_text = " ".join(participant_texts)
                texts.append(combined_text)
                labels.append(is_bot)
    
    print(f"📊 Created dataset with {len(texts)} samples")
    print(f"   Human samples: {labels.count(0)}")
    print(f"   Bot samples: {labels.count(1)}")
    
    # Split data 
    from sklearn.model_selection import train_test_split
    train_texts, val_texts, train_labels, val_labels = train_test_split(
        texts,
        labels,
        test_size=0.2,
        random_state=42,
        stratify=labels
    )
    
    # Log dataset information
    tracker.log_data_info(train_texts, train_labels, val_texts, val_labels)
    
    # Load model and tokenizer
    print("Loading model and tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(args.base_model)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    
    model = AutoModelForSequenceClassification.from_pretrained(
        args.base_model,
        num_labels=2
    )
    
    # LoRA configuration
    lora_config = LoraConfig(
        task_type=TaskType.SEQ_CLS,
        r=args.lora_r,
        lora_alpha=args.lora_alpha,
        lora_dropout=0.1,
        target_modules=["q_lin", "v_lin"] if "distilbert" in args.base_model else ["query", "value"]
    )
    
    # Apply LoRA
    model = get_peft_model(model, lora_config)
    model.print_trainable_parameters()
    
    # Log model information
    tracker.log_model_info(args.base_model, args.base_model, lora_config)
    
    # Prepare datasets
    def tokenize_function(examples):
        return tokenizer(examples["text"], truncation=True, padding="max_length", max_length=128)
    
    train_dataset = Dataset.from_dict({"text": train_texts, "labels": train_labels})
    val_dataset = Dataset.from_dict({"text": val_texts, "labels": val_labels})
    
    train_dataset = train_dataset.map(tokenize_function, batched=True)
    val_dataset = val_dataset.map(tokenize_function, batched=True)
    
    # Training arguments
    model_output_dir = tracker.output_dir / "model"
    training_args = TrainingArguments(
        output_dir=str(model_output_dir),
        learning_rate=args.learning_rate,
        per_device_train_batch_size=args.batch_size,
        per_device_eval_batch_size=args.batch_size,
        num_train_epochs=args.epochs,
        weight_decay=0.01,
        evaluation_strategy="epoch",
        save_strategy="epoch",
        logging_steps=50,
        load_best_model_at_end=True,
        metric_for_best_model="accuracy",
        greater_is_better=True,
        save_total_limit=2,
        report_to=[],  # Completely disable all automatic integrations
        disable_tqdm=False
    )
    
    # Log training configuration
    tracker.log_training_info(training_args)
    
    # Initialize trainer
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        compute_metrics=compute_metrics,
        callbacks=[EarlyStoppingCallback(early_stopping_patience=2)]
    )
    
    # Train model
    print("🔥 Starting training...")
    start_time = time.time()
    train_result = trainer.train()
    training_time = time.time() - start_time
    
    # Evaluate model
    print("Evaluating model...")
    eval_results = trainer.evaluate()
    
    # Log results
    tracker.log_training_results(trainer, eval_results)
    
    # Save model and get weights info
    print("Saving model...")
    trainer.save_model()
    
    # Get LoRA weights information
    trainable_params = model.get_nb_trainable_parameters()
    total_params = model.num_parameters()
    
    # Handle case where get_nb_trainable_parameters returns a tuple
    if isinstance(trainable_params, tuple):
        trainable_params = trainable_params[0]
    if isinstance(total_params, tuple):
        total_params = total_params[0]
    
    lora_weights_info = {
        "trainable_params": trainable_params,
        "total_params": total_params,
        "trainable_percentage": (trainable_params / total_params) * 100
    }
    
    # Log artifacts
    tracker.log_model_artifacts(model_output_dir, lora_weights_info, model)
    
    # Save experiment metadata and generate report
    tracker.save_experiment_metadata()
    tracker.generate_experiment_report()
    
    # End MLflow run
    tracker.end_run()
    
    print("\n🎉 Experiment Complete!")
    print(f"📁 All results saved to: {tracker.output_dir}")
    print(f"🎯 Final Validation Accuracy: {eval_results['eval_accuracy']:.4f}")
    print(f"⏱️ Total Training Time: {training_time:.2f} seconds")
    print(f"\n📊 View MLflow UI:")
    print(f"   Command: {tracker.get_mlflow_ui_command()}")
    print(f"   Then open: http://localhost:5000")

if __name__ == "__main__":
    main() 