import pandas as pd
from typing import Tuple, List
from abc import ABC, abstractmethod
import numpy as np

class BaseDetector(ABC):
    def __init__(self):
        self.feature_names = None
        self.train_data = None

    @abstractmethod
    def train(self, train_data: pd.DataFrame, feature_names: List[str]) -> None:
        pass

    @abstractmethod
    def predict(self, data: pd.DataFrame) -> Tuple[np.ndarray, List[List[str]]]:
        pass
