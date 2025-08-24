import pandas as pd
import numpy as np
from sklearn.neighbors import LocalOutlierFactor, NearestNeighbors
from typing import Tuple, List

from .base_detector import BaseDetector

class LOFDetector(BaseDetector):
    def __init__(self, n_neighbors: int = 40):
        super().__init__()
        self.n_neighbors = n_neighbors
        self.lof_model = None
        self.train_scores = None

    def train(self, train_data: pd.DataFrame, feature_names: List[str]) -> None:
        print("Training LOF model")
        self.feature_names = feature_names
        self.train_data = train_data.copy()

        self.lof_model = LocalOutlierFactor(
            n_neighbors=self.n_neighbors, contamination='auto', novelty=True
        )
        try:
            X_train = train_data[self.feature_names].values
        except KeyError as e:
            print(f"Error: One or more feature names not found in training data: {e}")
            raise
        
        try:
            self.lof_model.fit(X_train)
            self.train_scores = self.lof_model.score_samples(X_train)
        except Exception as e:
            print(f"An error occurred during model training: {e}")
            raise
            
        print(f"LOF training done")

    def predict(self, data: pd.DataFrame) -> Tuple[np.ndarray, List[List[str]]]:
        if self.lof_model is None:
            raise RuntimeError("The model has not been trained yet, please call the 'train' method before 'predict'")

        try:
            X_test = data[self.feature_names].values
            lof_scores = self.lof_model.score_samples(X_test)
        except KeyError as e:
            print(f"Error: One or more feature names not found in prediction data: {e}")
            raise
        except Exception as e:
            print(f"An error occurred during prediction: {e}")
            raise
            
        abnormality_scores = self._convert_to_abnormality_scores(lof_scores)
        contributors = self._get_contributors(data)
        return abnormality_scores, contributors

    def _convert_to_abnormality_scores(self, lof_scores: np.ndarray) -> np.ndarray:
        if self.train_scores is None:
            raise RuntimeError("Training scores are not available, the model may not have been trained correctly")
        
        try:
            train_min, train_max = self.train_scores.min(), self.train_scores.max()
            train_range = max(train_max - train_min, 1e-6)
            abnormality_scores = np.zeros_like(lof_scores, dtype=float)

            for i, score in enumerate(lof_scores):
                if score >= train_max:
                    abnormality_scores[i] = 0.0
                elif score >= train_min:
                    position = (score - train_min) / train_range
                    abnormality_scores[i] = 10 * (1 - position)
                else:
                    worst_test_score = lof_scores.min()
                    beyond_training = train_min - score
                    worst_beyond = max(train_min - worst_test_score, 1e-6)
                    relative_position = beyond_training / worst_beyond
                    abnormality_scores[i] = 10 + 90 * np.sqrt(relative_position)

            return np.clip(abnormality_scores, 0, 100)
        except Exception as e:
            print(f"An unexpected error occurred during abnormality score conversion: {e}")
            raise

    def _get_contributors(self, data: pd.DataFrame) -> List[List[str]]:
        if self.train_data is None:
            raise RuntimeError("Training data is not available, pllease train the model first")

        all_contributors = []
        try:
            data_matrix = data[self.feature_names].values
            train_matrix = self.train_data[self.feature_names].values

            nbrs = NearestNeighbors(n_neighbors=self.n_neighbors)
            nbrs.fit(train_matrix)

            for data_point in data_matrix:
                _, indices = nbrs.kneighbors([data_point])
                neighbors = train_matrix[indices[0]]
                feature_deviations = {}
                for i, feature_name in enumerate(self.feature_names):
                    neighbor_values = neighbors[:, i]
                    mean_neighbor, std_neighbor = np.mean(neighbor_values), np.std(neighbor_values) + 1e-6
                    deviation = abs(data_point[i] - mean_neighbor) / std_neighbor
                    feature_deviations[feature_name] = deviation

                sorted_contrib = sorted(feature_deviations.items(), key=lambda x: x[1], reverse=True)
                top_contributors = [feat for feat, contrib in sorted_contrib if contrib > 0.01][:7]
                all_contributors.append(top_contributors + [""] * (7 - len(top_contributors)))
            
            return all_contributors
        except KeyError as e:
            print(f"Error: One or more feature names not found in data for contributor calculation: {e}")
            raise
        except Exception as e:
            print(f"An unexpected error occurred while getting contributors: {e}")
            raise