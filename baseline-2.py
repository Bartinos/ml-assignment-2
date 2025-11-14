import numpy as np
import pandas as pd
from pathlib import Path
import json, zipfile

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler, LabelEncoder
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import f1_score, confusion_matrix


def create_submission(
    predicted_labels,
    ids=None,
    zip_name="output/predictions.zip",
    json_name="output/predictions.json",
    allowed_labels=("green", "yellow", "orange", "red"),
):
    """
    Create a CodaLab-compatible submission file.
    Saves a JSON list [{"id": int, "value": str}] and zips it.
    """
    if ids is None:
        ids = range(len(predicted_labels))
    ids = list(ids)

    if len(ids) != len(predicted_labels):
        raise ValueError("Length mismatch between ids and labels.")
    if len(set(ids)) != len(ids):
        raise ValueError("Duplicate IDs detected.")

    output = []
    for i, lab in zip(ids, predicted_labels):
        lab = str(lab)
        if allowed_labels and lab not in allowed_labels:
            raise ValueError(f"Invalid label: {lab}")
        output.append({"id": int(i), "value": lab})

    output.sort(key=lambda r: r["id"])
    Path(json_name).write_text(json.dumps(output, indent=2), encoding="utf-8")

    with zipfile.ZipFile(zip_name, "w", zipfile.ZIP_DEFLATED) as z:
        z.write(json_name, arcname=json_name)

    print(f"Saved: {json_name}, {zip_name}")
    return str(zip_name)


def calculate_resilience_cost(conf_matrix, cost_matrix):
    """
    Compute total Resilience Cost = sum(conf_matrix * cost_matrix).
    Lower is better (less costly misclassification).
    """
    if conf_matrix.shape != cost_matrix.shape:
        raise ValueError("Confusion and cost matrices must have the same shape.")
    return float(np.sum(conf_matrix * cost_matrix))


if __name__ == "__main__":
    """
    Baseline pipeline:
    1. Load train/test data and cost matrix
    2. Preprocess (encode, scale, split)
    3. Train a simple KNN classifier
    4. Evaluate F1 and Resilience Cost
    5. Generate submission.zip
    """
    SEED = 42
    np.random.seed(SEED)

    # --- Load data ---
    df = pd.read_csv("data/train.csv")
    test_df = pd.read_csv("data/test.csv")
    cost_matrix = pd.read_csv("data/cost_matrix.csv", index_col=0).values

    # --- Encode target and split ---
    enc = LabelEncoder().fit(df["alert"])
    y = enc.transform(df["alert"])
    X = df.drop("alert", axis=1)

    X_train, X_val, y_train, y_val = train_test_split(
        X, y, test_size=0.2, random_state=SEED, stratify=y
    )

    # --- Scale features ---
    scaler = MinMaxScaler().fit(X_train)
    X_train, X_val, X_test = (
        scaler.transform(X_train),
        scaler.transform(X_val),
        scaler.transform(test_df),
    )

    # --- Train baseline model ---
    model = KNeighborsClassifier(n_neighbors=10)
    model.fit(X_train, y_train)

    # --- Validate ---
    y_val_pred = model.predict(X_val)
    f1 = f1_score(y_val, y_val_pred, average="macro")
    cm = confusion_matrix(enc.inverse_transform(y_val), enc.inverse_transform(y_val_pred))
    rci = calculate_resilience_cost(cm, cost_matrix)

    print(f"F1 (macro): {f1:.3f}")
    print("Confusion matrix:\n", cm)
    print(f"Resilience Cost: {rci:.2f}")

    # --- Predict test and submit ---
    y_test_pred = enc.inverse_transform(model.predict(X_test))
    create_submission(y_test_pred)
