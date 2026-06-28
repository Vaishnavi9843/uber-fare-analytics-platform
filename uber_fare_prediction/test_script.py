from src.config import FEATURE_DATA_PATH

from src.preprocessing import (
    load_dataset,
    prepare_features,
    split_dataset,
)

from src.model import (
    train_random_forest,
    save_model,
    load_model,
)

from src.evaluate import (
    evaluate_regression_model,
    print_metrics,
)

print("=" * 60)
print("TESTING COMPLETE ML PIPELINE")
print("=" * 60)

# Load dataset
df = load_dataset(FEATURE_DATA_PATH)

# Prepare features
X, y = prepare_features(df)

# Split dataset
X_train, X_test, y_train, y_test = split_dataset(
    X,
    y
)

# Train model
model = train_random_forest(
    X_train,
    y_train
)

# Save model
save_model(model)

# Load model
loaded_model = load_model()

# Predict
pred = loaded_model.predict(X_test)

# Evaluate
metrics = evaluate_regression_model(
    y_test,
    pred
)

print_metrics(metrics)

print("\nPipeline executed successfully.")