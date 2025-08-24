import pandas as pd
import numpy as np
from typing import List

from src.detectors.lof_detector import LOFDetector
from src.detectors.mcd_detector import MCDDetector
from src.detectors.vae_detector import VAEDetector

class UnifiedAnomalyDetector:
    def __init__(self, lof_neighbors: int = 40, mcd_support: float = 0.8,
                 vae_latent_dim: int = 10, vae_epochs: int = 50):
        self.threshold_detector = LOFDetector(n_neighbors=lof_neighbors)
        self.relationship_detector = MCDDetector(support_fraction=mcd_support)
        self.pattern_detector = VAEDetector(latent_dim=vae_latent_dim, epochs=vae_epochs)
        self.feature_names = None

    def train(self, train_data: pd.DataFrame, feature_names: List[str]) -> None:
        print("Training")
        self.feature_names = feature_names
        self.threshold_detector.train(train_data, feature_names)
        self.relationship_detector.train(train_data, feature_names)
        self.pattern_detector.train(train_data, feature_names)
        print("\nAll detectors trained")

    def predict(self, data: pd.DataFrame) -> pd.DataFrame:
        threshold_scores, threshold_contrib = self.threshold_detector.predict(data)
        relationship_scores, relationship_contrib = self.relationship_detector.predict(data)
        pattern_scores, pattern_contrib = self.pattern_detector.predict(data)

        unified_scores, unified_contributors = [], []
        for i in range(len(data)):
            scores = [threshold_scores[i], relationship_scores[i], pattern_scores[i]]
            contributors = [threshold_contrib[i], relationship_contrib[i], pattern_contrib[i]]
            max_idx = np.argmax(scores)
            unified_scores.append(scores[max_idx])
            unified_contributors.append(contributors[max_idx])

        results = pd.DataFrame({'Abnormality_score': unified_scores}, index=data.index)
        for i in range(7):
            results[f'top_feature_{i+1}'] = [contrib[i] for contrib in unified_contributors]
        return results
