import pandas as pd
import numpy as np
from typing import Tuple, List
import warnings
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers

from .base_detector import BaseDetector

warnings.filterwarnings('ignore')
try:
    tf.config.threading.set_inter_op_parallelism_threads(0)
    tf.config.threading.set_intra_op_parallelism_threads(0)
except RuntimeError as e:
    print(f"Could not set TensorFlow threading options, they might have been set already: {e}")

np.random.seed(42)
tf.random.set_seed(42)

class VAEDetector(BaseDetector):
    def __init__(self, latent_dim: int = 8, epochs: int = 30, batch_size: int = 64):
        super().__init__()
        self.latent_dim = latent_dim
        self.epochs = epochs
        self.batch_size = batch_size
        self.autoencoder = None
        self.train_scores = None
        self.input_dim = None

    def _build_autoencoder(self):
        try:
            input_layer = keras.Input(shape=(self.input_dim,))
            encoded = layers.Dense(32, activation='relu')(input_layer)
            encoded = layers.Dense(16, activation='relu')(encoded)
            encoded = layers.Dense(self.latent_dim, activation='relu')(encoded)
            decoded = layers.Dense(16, activation='relu')(encoded)
            decoded = layers.Dense(32, activation='relu')(decoded)
            decoded = layers.Dense(self.input_dim, activation='linear')(decoded)
            self.autoencoder = keras.Model(input_layer, decoded)
            self.autoencoder.compile(optimizer='adam', loss='mse')
        except Exception as e:
            print(f"Failed to build the autoencoder model: {e}")
            raise

    def train(self, train_data: pd.DataFrame, feature_names: List[str]) -> None:
        print("Training Variational Autoencoder model")
        self.feature_names = feature_names
        self.train_data = train_data.copy()
        
        try:
            self.input_dim = len(feature_names)
            if self.input_dim == 0:
                raise ValueError("feature_names list cannot be empty")
            X_train = train_data[self.feature_names].values
        except KeyError as e:
            print(f"Error: One or more feature names not found in training data: {e}")
            raise
        except ValueError as e:
            print(f"Error during training data preparation: {e}")
            raise

        self._build_autoencoder()

        try:
            self.autoencoder.fit(
                X_train, X_train, epochs=self.epochs, batch_size=self.batch_size,
                validation_split=0.1, verbose=0, shuffle=True
            )
            X_train_pred = self.autoencoder.predict(X_train, verbose=0)
            self.train_scores = np.mean((X_train - X_train_pred) ** 2, axis=1)
        except Exception as e:
            print(f"An error occurred during model training or initial prediction: {e}")
            raise

        print("Variational Autoencoder training done")

    def predict(self, data: pd.DataFrame) -> Tuple[np.ndarray, List[List[str]]]:
        if self.autoencoder is None:
            raise RuntimeError("The model has not been trained yet, please call the 'train' method before 'predict'")

        try:
            X_data = data[self.feature_names].values
            X_reconstructed = self.autoencoder.predict(X_data, verbose=0)
        except KeyError as e:
            print(f"Error: One or more feature names not found in prediction data: {e}")
            raise
        except Exception as e:
            print(f"An error occurred during prediction: {e}")
            raise
        
        reconstruction_errors = np.mean((X_data - X_reconstructed) ** 2, axis=1)
        abnormality_scores = self._convert_to_abnormality_scores(reconstruction_errors)
        contributors = self._get_contributors(X_data, X_reconstructed)
        return abnormality_scores, contributors

    def _convert_to_abnormality_scores(self, reconstruction_errors: np.ndarray) -> np.ndarray:
        if self.train_scores is None:
            raise RuntimeError("Training scores are not available, the model may not have been trained correctly")

        try:
            train_max = max(self.train_scores.max(), 1e-6)
            cap_multiplier = 20.0
            log_cap = np.log(cap_multiplier)
            abnormality_scores = np.zeros_like(reconstruction_errors, dtype=float)

            for i, error in enumerate(reconstruction_errors):
                if error <= train_max:
                    abnormality_scores[i] = 10 * (error / train_max)
                else:
                    beyond_factor = error / train_max
                    scaled_log = np.log(beyond_factor) / log_cap
                    abnormality_scores[i] = 10 + 90 * scaled_log

            abnormality_scores = np.clip(abnormality_scores, 0, 100)
            noise = np.random.uniform(0.01, 0.1, size=abnormality_scores.shape)
            return np.clip(abnormality_scores + noise, 0, 100)
        except ZeroDivisionError:
            print("Error: Maximum training score is zero, cannot calculate abnormality scores ")
            return np.zeros_like(reconstruction_errors, dtype=float)
        except Exception as e:
            print(f"An unexpected error occurred during abnormality score conversion: {e}")
            raise

    def _get_contributors(self, original: np.ndarray, reconstructed: np.ndarray) -> List[List[str]]:
        if self.feature_names is None:
             raise RuntimeError("Feature names are not set, the model may not have been trained")
        
        all_contributors = []
        try:
            feature_errors = (original - reconstructed) ** 2
            for i in range(len(original)):
                contributions = {feat: feature_errors[i, j] for j, feat in enumerate(self.feature_names)}
                sorted_contrib = sorted(contributions.items(), key=lambda x: x[1], reverse=True)
                top_contributors = [feat for feat, contrib in sorted_contrib if contrib > 0.01][:7]
                all_contributors.append(top_contributors + [""] * (7 - len(top_contributors)))
            return all_contributors
        except Exception as e:
            print(f"An unexpected error occurred while getting contributors: {e}")
            raise