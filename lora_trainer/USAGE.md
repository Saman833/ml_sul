# LoRA Bot Detection - Usage Guide

## Quick Start

### 1. Run Training Experiment
```bash
# Basic training
python experiment_tracker.py --experiment_name "bot_detection_v1" --epochs 3

# Custom parameters
python experiment_tracker.py --experiment_name "high_performance" --lora_r 32 --lora_alpha 64 --learning_rate 0.0002 --batch_size 16 --epochs 3
```

### 2. Use Experiment Runner (Presets)
```bash
python run_experiment.py
# Choose: 1 (baseline), 2 (high rank), 3 (low rank), 4 (quick test), or 5 (custom)
```

### 3. View MLflow Results
```bash
python view_experiments.py
# Choose: 1 (list), 2 (MLflow UI), 3 (compare), 4 (exit)
```

### 4. Test Trained Model
```bash
python test_bot_detector.py
```

## MLflow UI
```bash
mlflow ui --backend-store-uri "file:///path/to/trainer/experiments/your_experiment/mlruns"
# Then open: http://localhost:5000
```

## Best Parameters
- **Learning Rate**: 0.0003
- **LoRA Rank**: 16-32
- **Batch Size**: 16
- **Epochs**: 3 