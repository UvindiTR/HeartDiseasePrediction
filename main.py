import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import joblib
import tensorflow as tf

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (
    confusion_matrix,
    classification_report,
    ConfusionMatrixDisplay,
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score
)

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Input, Dropout, BatchNormalization
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
from tensorflow.keras.optimizers import Adam


# ============================================================
# 0. REPRODUCIBILITY
# ============================================================

np.random.seed(42)
tf.random.set_seed(42)


# ============================================================
# 1. LOAD DATASET
# ============================================================

df = pd.read_csv("heart_disease.csv")

print("Dataset loaded successfully!")
print("Dataset Shape:", df.shape)


# ============================================================
# 2. DATASET INFORMATION
# ============================================================

print("\n========== Dataset Information ==========")
df.info()

print("\n========== Missing Values ==========")
print(df.isnull().sum())

print("\n========== Duplicate Rows ==========")
print("Duplicate rows:", df.duplicated().sum())


# ============================================================
# 3. REMOVE UNNECESSARY COLUMN
# ============================================================

# 'num' is not used for prediction
if "num" in df.columns:
    df = df.drop("num", axis=1)

print("\n'num' column removed.")


# ============================================================
# 4. SEPARATE FEATURES AND TARGET
# ============================================================

X = df.drop("target_binary", axis=1)
y = df["target_binary"]


print("\n========== Features ==========")
print(X.columns.tolist())

print("\nNumber of input features:", X.shape[1])


# ============================================================
# 5. TARGET DISTRIBUTION
# ============================================================

print("\n========== Target Distribution ==========")
print(y.value_counts())

print("\nTarget Percentage:")
print(y.value_counts(normalize=True) * 100)


# ============================================================
# 6. TRAIN / VALIDATION / TEST SPLIT
# ============================================================

# 80% temporary + 20% test
X_temp, X_test, y_temp, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=42,
    stratify=y
)

# From the remaining 80%:
# 80% training + 20% validation
X_train, X_val, y_train, y_val = train_test_split(
    X_temp,
    y_temp,
    test_size=0.20,
    random_state=42,
    stratify=y_temp
)


print("\n========== Dataset Split ==========")

print("Training Data:", X_train.shape)
print("Validation Data:", X_val.shape)
print("Testing Data:", X_test.shape)


# ============================================================
# 7. DATA PREPROCESSING
# ============================================================

# StandardScaler keeps the original 13 input features.
# This makes the preprocessing simple and compatible
# with the GUI.

scaler = StandardScaler()

X_train_processed = scaler.fit_transform(X_train)

X_val_processed = scaler.transform(X_val)

X_test_processed = scaler.transform(X_test)


print("\n========== Preprocessing ==========")

print(
    "Original number of features:",
    X.shape[1]
)

print(
    "Processed training shape:",
    X_train_processed.shape
)

print(
    "Processed validation shape:",
    X_val_processed.shape
)

print(
    "Processed testing shape:",
    X_test_processed.shape
)


# ============================================================
# 8. BUILD NEURAL NETWORK
# ============================================================

input_features = X_train_processed.shape[1]

model = Sequential([

    Input(shape=(input_features,)),

    Dense(64, activation="relu"),
    BatchNormalization(),
    Dropout(0.20),

    Dense(32, activation="relu"),
    BatchNormalization(),
    Dropout(0.15),

    Dense(16, activation="relu"),

    Dense(1, activation="sigmoid")
])


# ============================================================
# 9. COMPILE MODEL
# ============================================================

model.compile(

    optimizer=Adam(
        learning_rate=0.001
    ),

    loss="binary_crossentropy",

    metrics=[
        "accuracy",

        tf.keras.metrics.AUC(
            name="auc"
        ),

        tf.keras.metrics.Precision(
            name="precision"
        ),

        tf.keras.metrics.Recall(
            name="recall"
        )
    ]
)


print("\nModel compiled successfully!")

model.summary()


# ============================================================
# 10. CALLBACKS
# ============================================================

early_stopping = EarlyStopping(

    monitor="val_auc",

    mode="max",

    patience=15,

    restore_best_weights=True,

    verbose=1
)


reduce_lr = ReduceLROnPlateau(

    monitor="val_loss",

    factor=0.5,

    patience=5,

    min_lr=0.00001,

    verbose=1
)


# ============================================================
# 11. TRAIN NEURAL NETWORK
# ============================================================

print("\n========== Training Neural Network ==========")

history = model.fit(

    X_train_processed,

    y_train,

    validation_data=(
        X_val_processed,
        y_val
    ),

    epochs=150,

    batch_size=32,

    callbacks=[
        early_stopping,
        reduce_lr
    ],

    verbose=1
)


# ============================================================
# 12. TRAINING RESULTS
# ============================================================

print("\n========== Final Training Results ==========")

print(
    "Training Accuracy:",
    round(history.history["accuracy"][-1], 4)
)

print(
    "Validation Accuracy:",
    round(history.history["val_accuracy"][-1], 4)
)

print(
    "Training AUC:",
    round(history.history["auc"][-1], 4)
)

print(
    "Validation AUC:",
    round(history.history["val_auc"][-1], 4)
)

print(
    "Training Loss:",
    round(history.history["loss"][-1], 4)
)

print(
    "Validation Loss:",
    round(history.history["val_loss"][-1], 4)
)

print(
    "Number of Epochs Completed:",
    len(history.history["loss"])
)


# ============================================================
# 13. SAVE SCALER
# ============================================================

joblib.dump(
    scaler,
    "scaler.pkl"
)

print("\nScaler saved successfully!")


# ============================================================
# 14. SAVE MODEL
# ============================================================

model.save(
    "heart_model.keras"
)

print("Model saved successfully!")


# ============================================================
# 15. TEST MODEL
# ============================================================

print("\n========== Test Results ==========")

test_results = model.evaluate(
    X_test_processed,
    y_test,
    verbose=0
)

for name, value in zip(
    model.metrics_names,
    test_results
):

    print(
        name + ":",
        round(float(value), 4)
    )


# ============================================================
# 16. TEST SET PREDICTIONS
# ============================================================

test_probabilities = model.predict(
    X_test_processed,
    verbose=0
).flatten()


# ============================================================
# 17. USE STANDARD CLASSIFICATION THRESHOLD
# ============================================================

threshold = 0.50

y_pred = (
    test_probabilities >= threshold
).astype(int)


print("\nClassification Threshold:", threshold)


# ============================================================
# 18. FINAL PERFORMANCE
# ============================================================

accuracy = accuracy_score(
    y_test,
    y_pred
)

precision = precision_score(
    y_test,
    y_pred,
    zero_division=0
)

recall = recall_score(
    y_test,
    y_pred,
    zero_division=0
)

f1 = f1_score(
    y_test,
    y_pred,
    zero_division=0
)

auc = roc_auc_score(
    y_test,
    test_probabilities
)


print("\n========== FINAL TEST PERFORMANCE ==========")

print(
    "Accuracy:",
    round(accuracy, 4)
)

print(
    "Accuracy (%):",
    round(accuracy * 100, 2)
)

print(
    "Precision:",
    round(precision, 4)
)

print(
    "Recall:",
    round(recall, 4)
)

print(
    "F1 Score:",
    round(f1, 4)
)

print(
    "AUC:",
    round(auc, 4)
)


# ============================================================
# 19. CONFUSION MATRIX
# ============================================================

cm = confusion_matrix(
    y_test,
    y_pred
)

print("\n========== Confusion Matrix ==========")

print(cm)


# ============================================================
# 20. CLASSIFICATION REPORT
# ============================================================

print("\n========== Classification Report ==========")

print(
    classification_report(
        y_test,
        y_pred,
        target_names=[
            "No Heart Disease",
            "Heart Disease"
        ],
        zero_division=0
    )
)


# ============================================================
# 21. EXAMPLE PATIENT PREDICTION
# ============================================================

probability = test_probabilities[0]

print("\n========== Example Prediction ==========")

print(
    "Prediction Probability:",
    round(float(probability), 4)
)

if probability >= threshold:

    print(
        "Prediction: Heart Disease"
    )

else:

    print(
        "Prediction: No Heart Disease"
    )

print(
    "Actual Result:",
    y_test.iloc[0]
)


# ============================================================
# 22. DISPLAY CONFUSION MATRIX
# ============================================================

disp = ConfusionMatrixDisplay(

    confusion_matrix=cm,

    display_labels=[
        "No Heart Disease",
        "Heart Disease"
    ]
)

disp.plot()

plt.title(
    "Heart Disease Prediction - Confusion Matrix"
)

plt.show()


# ============================================================
# 23. TRAINING / VALIDATION ACCURACY GRAPH
# ============================================================

plt.figure(figsize=(8, 5))

plt.plot(
    history.history["accuracy"],
    label="Training Accuracy"
)

plt.plot(
    history.history["val_accuracy"],
    label="Validation Accuracy"
)

plt.title(
    "Training and Validation Accuracy"
)

plt.xlabel("Epoch")

plt.ylabel("Accuracy")

plt.legend()

plt.grid(True)

plt.show()


# ============================================================
# 24. TRAINING / VALIDATION LOSS GRAPH
# ============================================================

plt.figure(figsize=(8, 5))

plt.plot(
    history.history["loss"],
    label="Training Loss"
)

plt.plot(
    history.history["val_loss"],
    label="Validation Loss"
)

plt.title(
    "Training and Validation Loss"
)

plt.xlabel("Epoch")

plt.ylabel("Loss")

plt.legend()

plt.grid(True)

plt.show()


# ============================================================
# 25. AUC GRAPH
# ============================================================

plt.figure(figsize=(8, 5))

plt.plot(
    history.history["auc"],
    label="Training AUC"
)

plt.plot(
    history.history["val_auc"],
    label="Validation AUC"
)

plt.title(
    "Training and Validation AUC"
)

plt.xlabel("Epoch")

plt.ylabel("AUC")

plt.legend()

plt.grid(True)

plt.show()