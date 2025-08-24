import pandas as pd
import numpy as np
from sklearn.covariance import MinCovDet
from typing import Tuple, List

from .base_detector import BaseDetector

class MCDDetector(BaseDetector):
    def __init__(self, support_fraction: float = 0.8):
        super().__init__()
        self.support_fraction = support_fraction
        self.mcd_model = None
        self.train_scores = None

    def train(self, train_data: pd.DataFrame, feature_names: List[str]) -> None:
        print("Training MCD model")
        self.feature_names = feature_names
        self.train_data = train_data.copy()

        self.mcd_model = MinCovDet(support_fraction=self.support_fraction, random_state=42)
        try:
            X_train = train_data[self.feature_names].values
        except KeyError as e:
            print(f"Error: One or more feature names not found in training data: {e}")
            raise

        try:
            self.mcd_model.fit(X_train)
            self.train_scores = self.mcd_model.mahalanobis(X_train)
        except Exception as e:
            print(f"An error occurred during model training: {e}")
            raise
        
        print(f"MCD training done")

    def predict(self, data: pd.DataFrame) -> Tuple[np.ndarray, List[List[str]]]:
        if self.mcd_model is None:
            raise RuntimeError("The model has not been trained yet")

        try:
            X_data = data[self.feature_names].values
            mahal_distances = self.mcd_model.mahalanobis(X_data)
        except KeyError as e:
            print(f"Error: One or more feature names not found in prediction data: {e}")
            raise
        except Exception as e:
            print(f"An error occurred during prediction: {e}")
            raise
            
        abnormality_scores = self._convert_to_abnormality_scores(mahal_distances)
        contributors = self._get_contributors(data)
        return abnormality_scores, contributors

    def _convert_to_abnormality_scores(self, mahal_distances: np.ndarray) -> np.ndarray:
        if self.train_scores is None:
            raise RuntimeError("Training scores are not available, the model may not have been trained correctly")

        try:
            train_max = max(self.train_scores.max(), 1e-6)
            cap_multiplier = 20.0
            log_cap = np.log(cap_multiplier)
            abnormality_scores = np.zeros_like(mahal_distances, dtype=float)

            for i, distance in enumerate(mahal_distances):
                if distance <= train_max:
                    abnormality_scores[i] = 10 * (distance / train_max)
                else:
                    beyond_factor = distance / train_max
                    scaled_log = np.log(beyond_factor) / log_cap
                    abnormality_scores[i] = 10 + 90 * scaled_log

            abnormality_scores = np.clip(abnormality_scores, 0, 100)
            noise = np.random.uniform(0.01, 0.1, size=abnormality_scores.shape)
            return np.clip(abnormality_scores + noise, 0, 100)
        except ZeroDivisionError:
            print("Error: Maximum training score is zero, cannot calculate abnormality scores")
            return np.zeros_like(mahal_distances, dtype=float)
        except Exception as e:
            print(f"An unexpected error occurred during abnormality score conversion: {e}")
            raise

    def _get_contributors(self, data: pd.DataFrame) -> List[List[str]]:
        if not hasattr(self.mcd_model, 'precision_') or not hasattr(self.mcd_model, 'location_'):
            raise RuntimeError("The MCD model is not fitted, cannot get top contributors")

        all_contributors = []
        try:
            precision_matrix = self.mcd_model.precision_
            location = self.mcd_model.location_
            
            X_data = data[self.feature_names].values
            diff = X_data - location
            
            #vectorized contribution calculation
            term1 = np.dot(diff, precision_matrix)
            contributions_matrix = diff * term1
            
            for i in range(len(data)):
                contributions = {feat: abs(contributions_matrix[i, j]) for j, feat in enumerate(self.feature_names)}
                sorted_contrib = sorted(contributions.items(), key=lambda x: x[1], reverse=True)
                top_contributors = [feat for feat, contrib in sorted_contrib if contrib > 0.01][:7]
                all_contributors.append(top_contributors + [""] * (7 - len(top_contributors)))
                
            return all_contributors
        except KeyError as e:
            print(f"Error: One or more feature names not found in data for contributor calculation: {e}")
            raise
        except Exception as e:
            print(f"An unexpected error occurred while getting contributors: {e}")
            raise