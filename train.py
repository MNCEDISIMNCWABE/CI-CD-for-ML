import pandas as pd
import matplotlib.pyplot as plt
import skops.io as sio
from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OrdinalEncoder, StandardScaler
from sklearn.metrics import accuracy_score, f1_score, confusion_matrix, ConfusionMatrixDisplay
import os
from datetime import datetime
import glob
import re
import pytz

def read_data(file_path):
    """Read CSV data from the given file path."""
    df = pd.read_csv(file_path)
    df = df.sample(frac=1)  # Shuffle the dataset
    return df

def transform_data(cat_col, num_col):
    """Transform data by handling categorical and numerical features."""
    transform = ColumnTransformer([
        ("encoder", OrdinalEncoder(), cat_col),
        ("num_imputer", SimpleImputer(strategy="median"), num_col),
        ("num_scaler", StandardScaler(), num_col),
    ])
    return transform

def split_data(df, target_column, test_size=0.3, random_state=125):
    """Split the dataset into training and testing sets."""
    X = df.drop(target_column, axis=1).values
    y = df[target_column].values
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=random_state)
    return X_train, X_test, y_train, y_test

def pipeline(transform):
    """Create a machine learning pipeline."""
    pipe = Pipeline(steps=[
        ("preprocessing", transform),
        ("model", RandomForestClassifier(n_estimators=100, random_state=125)),
    ])
    return pipe

def train_and_evaluate(pipe, X_train, y_train, X_test, y_test):
    """Train the model and evaluate its performance."""
    pipe.fit(X_train, y_train)
    predictions = pipe.predict(X_test)
    accuracy = accuracy_score(y_test, predictions)
    f1 = f1_score(y_test, predictions, average="macro")
    return accuracy, f1, predictions

def get_timestamp():
    """Get current timestamp in SAST."""
    sast = pytz.timezone('Africa/Johannesburg')
    return datetime.now(sast).strftime("%Y%m%d_%H%M%S")

def plot_confusion_matrix(y_test, predictions, pipe, filepath="results"):
    """Plot and save confusion matrix with timestamp."""
    os.makedirs(filepath, exist_ok=True)
    ts = get_timestamp()
    full_path = os.path.join(filepath, f"confusion_matrix_{ts}.png")
    cm = confusion_matrix(y_test, predictions, labels=pipe.classes_)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=pipe.classes_)
    disp.plot()
    plt.savefig(full_path, dpi=120)
    plt.close()
    return full_path

def save_metrics(accuracy, f1, filepath="results"):
    """Save accuracy and F1 score with timestamp."""
    os.makedirs(filepath, exist_ok=True)
    ts = get_timestamp()
    full_path = os.path.join(filepath, f"metrics_{ts}.txt")
    with open(full_path, "w") as outfile:
        outfile.write(f"Accuracy = {round(accuracy, 2)}, F1 Score = {round(f1, 2)}")
    return full_path

def save_model(pipe, filepath="model"):
    """Save the trained model pipeline with timestamp."""
    os.makedirs(filepath, exist_ok=True)
    ts = get_timestamp()
    full_path = os.path.join(filepath, f"drug_pipeline_{ts}.skops")
    sio.dump(pipe, full_path)
    return full_path

def plot_performance_trend(filepath="results"):
    """Plot and save performance trend (accuracy and F1 score) over time."""
    os.makedirs(filepath, exist_ok=True)
    
    # Collect all metric files
    metric_files = glob.glob(os.path.join(filepath, "metrics_*.txt"))
    if not metric_files:
        print("No metric files found.")
        return None
    
    # Extract metrics and timestamps
    timestamps = []
    accuracies = []
    f1_scores = []
    
    for file in sorted(metric_files):
        with open(file, "r") as f:
            content = f.read()
            # Extract accuracy and F1 score using regex
            acc_match = re.search(r"Accuracy = (\d+\.\d+)", content)
            f1_match = re.search(r"F1 Score = (\d+\.\d+)", content)
            if acc_match and f1_match:
                accuracies.append(float(acc_match.group(1)))
                f1_scores.append(float(f1_match.group(1)))
                # Extract timestamp from filename
                ts = os.path.basename(file).replace("metrics_", "").replace(".txt", "")
                timestamps.append(datetime.strptime(ts, "%Y%m%d_%H%M%S"))
    
    if not timestamps:
        print("No valid metrics found for plotting.")
        return None
    
    # Create a line chart using Chart.js
    chart = {
        "type": "line",
        "data": {
            "labels": [ts.strftime("%Y-%m-%d %H:%M:%S") for ts in timestamps],
            "datasets": [
                {
                    "label": "Accuracy",
                    "data": accuracies,
                    "borderColor": "#4CAF50",
                    "backgroundColor": "rgba(76, 175, 80, 0.2)",
                    "fill": False,
                    "tension": 0.1
                },
                {
                    "label": "F1 Score",
                    "data": f1_scores,
                    "borderColor": "#2196F3",
                    "backgroundColor": "rgba(33, 150, 243, 0.2)",
                    "fill": False,
                    "tension": 0.1
                }
            ]
        },
        "options": {
            "responsive": True,
            "plugins": {
                "legend": {
                    "position": "top"
                },
                "title": {
                    "display": True,
                    "text": "Model Performance Trend Over Time"
                }
            },
            "scales": {
                "x": {
                    "title": {
                        "display": True,
                        "text": "Training Run (Timestamp)"
                    }
                },
                "y": {
                    "title": {
                        "display": True,
                        "text": "Score"
                    },
                    "min": 0,
                    "max": 1
                }
            }
        }
    }
    
    # Save the chart (for reference, assuming it's visualized in a UI or external tool)
    ts = get_timestamp()
    chart_path = os.path.join(filepath, f"performance_trend_{ts}.json")
    with open(chart_path, "w") as f:
        import json
        json.dump(chart, f)
    
    # Also create a matplotlib plot for immediate visualization
    plt.figure(figsize=(10, 6))
    plt.plot(timestamps, accuracies, label="Accuracy", color="#4CAF50", marker="o")
    plt.plot(timestamps, f1_scores, label="F1 Score", color="#2196F3", marker="o")
    plt.xlabel("Training Run (Timestamp)")
    plt.ylabel("Score")
    plt.title("Model Performance Trend Over Time")
    plt.legend()
    plt.xticks(rotation=45)
    plt.tight_layout()
    chart_png_path = os.path.join(filepath, f"performance_trend_{ts}.png")
    plt.savefig(chart_png_path, dpi=120)
    plt.close()
    
    return chart_path, chart_png_path

if __name__ == "__main__":
    # Load, transform and split data
    df = read_data("data/drug.csv")
    cat_col = [1, 2, 3]  # selects second, third and fourth columns for categorical encoding
    num_col = [0, 4]  # selects first and fifth columns for numerical processing
    transform = transform_data(cat_col, num_col)
    X_train, X_test, y_train, y_test = split_data(df, target_column="Drug")
    pipe = pipeline(transform)

    # Train the model and evaluate
    accuracy, f1, predictions = train_and_evaluate(pipe, X_train, y_train, X_test, y_test)

    # Save metrics, confusion matrix, and model
    save_metrics(accuracy, f1)
    plot_confusion_matrix(y_test, predictions, pipe)
    save_model(pipe)

    # Plot performance trend
    plot_performance_trend()
