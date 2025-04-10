from ucla_admissions.dataset import load_data
from ucla_admissions.features import preprocess_training_data
from ucla_admissions.modeling.train import load_or_train_model, evaluate_model
from sklearn.model_selection import train_test_split
from ucla_admissions.config import TEST_SIZE, RANDOM_STATE


# Load and preprocess data
df = load_data()
X_scaled, y, scaler, feature_columns = preprocess_training_data(df)

# Split into training and test sets (since X_scaled and y are now fully preprocessed)
X_train, X_test, y_train, y_test = train_test_split(
    X_scaled, y, test_size=TEST_SIZE, random_state=RANDOM_STATE, stratify=y
)

# Train or load model
model = load_or_train_model(X_train, y_train)

# Evaluate
accuracy, conf_matrix = evaluate_model(model, X_test, y_test)

# Display results
print(f"✅ Model Accuracy: {accuracy * 100:.2f}%")
print("🧮 Confusion Matrix:")
print(conf_matrix)
