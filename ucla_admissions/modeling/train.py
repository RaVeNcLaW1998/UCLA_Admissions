from sklearn.neural_network import MLPClassifier
from sklearn.metrics import accuracy_score, confusion_matrix
import joblib
import logging
import os
from ..config import MODEL_PATH, RANDOM_STATE

logger = logging.getLogger(__name__)


def train_model(X_train, y_train):
    try:
        mlp = MLPClassifier(
            hidden_layer_sizes=(3, 3),
            batch_size=50,
            max_iter=200,
            random_state=RANDOM_STATE,
        )
        mlp.fit(X_train, y_train)

        # Ensure the model directory exists
        model_dir = os.path.dirname(MODEL_PATH)
        if not os.path.exists(model_dir):
            os.makedirs(model_dir)

        joblib.dump(mlp, MODEL_PATH)
        logger.info("Model training completed and saved successfully.")
        return mlp
    except Exception as e:
        logger.error(f"Error during model training: {e}")
        raise


def evaluate_model(model, X_test, y_test):
    try:
        predictions = model.predict(X_test)
        acc = accuracy_score(y_test, predictions)
        conf_matrix = confusion_matrix(y_test, predictions)
        logger.info(f"Model evaluation accuracy: {acc}")
        return acc, conf_matrix
    except Exception as e:
        logger.error(f"Error during model evaluation: {e}")
        raise


def load_or_train_model(X_train, y_train):
    try:
        if os.path.exists(MODEL_PATH):
            model = joblib.load(MODEL_PATH)
            logger.info("Loaded existing model.")
        else:
            logger.info("No existing model found. Training a new model.")
            model = train_model(X_train, y_train)
        return model
    except Exception as e:
        logger.error(f"Error loading or training model: {e}")
        raise
