# Include plotting functions from previous messages
import matplotlib.pyplot as plt
import logging

logger = logging.getLogger(__name__)


def plot_loss_curve(model):
    try:
        fig = plt.figure(figsize=(10, 6))
        plt.plot(model.loss_curve_, label="Loss", color="blue")
        plt.title("Loss Curve")
        plt.xlabel("Iterations")
        plt.ylabel("Loss")
        plt.legend()
        plt.grid(True)
        logger.info("Loss curve plotted successfully.")
        return fig
    except Exception as e:
        logger.error(f"Error plotting loss curve: {e}")
        raise
