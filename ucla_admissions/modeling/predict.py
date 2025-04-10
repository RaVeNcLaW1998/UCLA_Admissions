import logging

logger = logging.getLogger(__name__)


def predict_admission(X_new, model):
    try:
        prediction = model.predict(X_new)
        probability = model.predict_proba(X_new)[:, 1]
        logger.info("Prediction made successfully.")
        return prediction, probability

    except Exception as e:
        logger.error(f"Error during prediction: {e}")
        raise
