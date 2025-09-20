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

def read_data(file_path):
    """Read CSV data from the given file path."""
    df = pd.read_csv(file_path)
    df = df.sample(frac=1)  # Shuffle the dataset
    return df

def transform_data(cat_col, num_col):
    """Transform data by handling categorical and numerical features."""
   # Define preprocessing
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
    """Get current timestamp"""
    return datetime.now().strftime("%Y%m%d_%H%M%S")

def plot_confusion_matrix(y_test, predictions, pipe, filepath="results"):
    """Plot and save confusion matrix with timestamp."""
    os.makedirs(filepath, exist_ok=True)
    ts = get_timestamp()
    full_path = os.path.join(filepath, f"confusion_matrix_{ts}.png")
    cm = confusion_matrix(y_test, predictions, labels=pipe.classes_)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=pipe.classes_)
    disp.plot()
    plt.savefig(full_path, dpi=120)
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

    # Save metrics and model
    save_metrics(accuracy, f1)
    plot_confusion_matrix(y_test, predictions, pipe)
    save_model(pipe)