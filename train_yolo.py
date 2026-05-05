# ==========================================
# YOLO Training Pipeline for PPE Detection
# ==========================================

from pathlib import Path
import torch
import time
import pandas as pd
import matplotlib.pyplot as plt
from ultralytics import YOLO

# ==========================================
# CONFIGURATION
# ==========================================
DATASET_PATH = Path("./data")
OUTPUT_DIR = Path("./outputs")

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

MODELS = [
    "yolo26n.pt",
    "yolo26s.pt",
    "yolo26m.pt",
    "yolo26l.pt",
    "yolo26x.pt"
]

# ==========================================
# DATASET VALIDATION
# ==========================================
def validate_dataset():
    required_paths = [
        DATASET_PATH / "train/images",
        DATASET_PATH / "valid/images"
    ]

    for path in required_paths:
        if not path.exists():
            raise FileNotFoundError(f"Missing required directory: {path}")

    print("Dataset structure validated.")

# ==========================================
# YAML GENERATION
# ==========================================
def create_dataset_yaml():
    yaml_path = DATASET_PATH / "data.yaml"

    yaml_content = f"""
path: {DATASET_PATH}

train: train/images
val: valid/images
test: test/images

names:
  0: class_0
  1: class_1
  2: class_2
  3: class_3
  4: class_4
  5: class_5
  6: class_6
"""

    with open(yaml_path, "w") as f:
        f.write(yaml_content)

    return yaml_path

# ==========================================
# TRAINING FUNCTION
# ==========================================
def train_models(yaml_path):
    device = 0 if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")

    results = []

    for model_name in MODELS:
        print(f"\nTraining model: {model_name}")

        try:
            model = YOLO(model_name)
        except:
            print(f"Model {model_name} not found. Using fallback (yolov8n.pt).")
            model = YOLO("yolov8n.pt")
            model_name = "yolov8n.pt"

        run_name = model_name.replace(".pt", "")

        start_time = time.time()

        model.train(
            data=str(yaml_path),
            epochs=50,
            imgsz=640,
            batch=16,
            device=device,
            project=str(OUTPUT_DIR),
            name=run_name,
            exist_ok=True,
            workers=2
        )

        training_time = (time.time() - start_time) / 3600

        metrics = model.val()

        results.append({
            "Model": model_name,
            "mAP@0.5": float(metrics.box.map50),
            "mAP@0.5:0.95": float(metrics.box.map),
            "Precision": float(metrics.box.mp),
            "Recall": float(metrics.box.mr),
            "Training_Time_h": round(training_time, 2)
        })

    return results

# ==========================================
# SAVE RESULTS
# ==========================================
def save_results(results):
    df = pd.DataFrame(results)

    output_file = OUTPUT_DIR / "results.csv"
    df.to_csv(output_file, index=False)

    print("\nResults:")
    print(df)

    return df

# ==========================================
# PLOT RESULTS
# ==========================================
def plot_results(df):
    models = df["Model"]

    def plot(metric, filename):
        plt.figure()
        plt.plot(models, df[metric], marker='o')
        plt.xticks(rotation=45)
        plt.grid()
        plt.tight_layout()
        plt.savefig(OUTPUT_DIR / filename)
        plt.close()

    plot("mAP@0.5", "map50.png")
    plot("mAP@0.5:0.95", "map50_95.png")
    plot("Training_Time_h", "training_time.png")

# ==========================================
# MAIN
# ==========================================
def main():
    validate_dataset()
    yaml_path = create_dataset_yaml()

    results = train_models(yaml_path)
    df = save_results(results)

    plot_results(df)

    print("\nPipeline executed successfully.")

# ==========================================
# ENTRY POINT
# ==========================================
if __name__ == "__main__":
    main()
