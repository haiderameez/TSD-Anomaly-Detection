import pandas as pd
import numpy as np
from sklearn.preprocessing import RobustScaler
from typing import Tuple, List
import os

class DataPreprocessor:

    def __init__(self, file_path: str):
        self.file_path = file_path
        self.df = None
        self.feature_columns = None
        self.inferred_freq = None

    def load_and_prepare_data(self) -> None:
        print(f"Loading data from {self.file_path}")
        if not os.path.exists(self.file_path):
            raise FileNotFoundError(f"Error: The file '{self.file_path}' was not found.")

        self.df = pd.read_csv(self.file_path)

        if 'Time' not in self.df.columns:
            raise ValueError("Error: 'Time' column not found in the CSV file.")

        self.df['Time'] = pd.to_datetime(self.df['Time'])
        self.df.set_index('Time', inplace=True)
        self.df.sort_index(inplace=True) 

    def clean_and_validate_data(self) -> None:
        if self.df is None:
            raise ValueError("Data has not been loaded. Run load_and_prepare_data() first.")

        self.inferred_freq = pd.infer_freq(self.df.index[:5000])
        if self.inferred_freq is None:
            median_diff = self.df.index.to_series().diff().median()
            raise ValueError(f"Error: Timestamps are not at a regular interval. Median interval detected: {median_diff}.")
        
        expected_range = pd.date_range(start=self.df.index.min(), end=self.df.index.max(), freq=self.inferred_freq)
        if len(self.df.index) != len(expected_range):
            missing_timestamps = len(expected_range) - len(self.df.index)
            print(f"   - Warning: Found and filled {missing_timestamps} missing timestamps in the index.")
            self.df = self.df.reindex(expected_range)

        initial_nan_count = self.df.isna().sum().sum()
        
        for col in self.df.columns:
            self.df[col] = pd.to_numeric(self.df[col], errors='coerce')
            
        final_nan_count = self.df.isna().sum().sum()
        coerced_count = final_nan_count - initial_nan_count
        
        if coerced_count > 0:
            print(f"   - Found and replaced {coerced_count} non-numeric value(s) with NaN.")


        total_missing = self.df.isna().sum().sum()
        if total_missing > 0:
            print(f"   - Found {total_missing} missing value(s).")
            self.df.fillna(method='ffill', inplace=True)
            self.df.fillna(method='bfill', inplace=True)
            
            remaining_nans = self.df.isna().sum().sum()
            if remaining_nans > 0:
                raise ValueError(f"Error: Could not fill all missing values. {remaining_nans} NaNs remain. Check for completely empty columns.")

    def remove_constant_features(self, train_df: pd.DataFrame) -> None:
        constant_features = train_df.columns[train_df.var() == 0].tolist()

        if constant_features:
            print(f"   - Found {len(constant_features)} constant features to drop: {constant_features}")
            self.df.drop(columns=constant_features, inplace=True)

    def remove_correlated_features(self, train_df: pd.DataFrame, threshold: float = 0.95) -> List[str]:
        train_df_filtered = train_df[self.df.columns]
        
        corr_matrix = train_df_filtered.corr().abs()
        upper = corr_matrix.where(np.triu(np.ones(corr_matrix.shape), k=1).astype(bool))
        to_drop = [column for column in upper.columns if any(upper[column] > threshold)]

        if to_drop:
            print(f"   - Found {len(to_drop)} columns to drop: {to_drop}")
            self.df.drop(columns=to_drop, inplace=True)

        self.feature_columns = self.df.columns.tolist()
        print(f"Number of features remaining: {len(self.feature_columns)}")
        return self.feature_columns

    def split_data(self, train_start: str, train_end: str, analysis_end: str) -> Tuple[pd.DataFrame, pd.DataFrame]:
        train_df = self.df.loc[train_start:train_end]
        analysis_df = self.df.loc[train_start:analysis_end]
        try:
            time_step_duration = pd.Timedelta(pd.tseries.frequencies.to_offset(self.inferred_freq))
            
            points_per_hour = pd.Timedelta('1 hour') / time_step_duration
            
            min_required_points = 72 * points_per_hour

            if len(train_df) < min_required_points:
                 print(f"   - Warning: Training data has {len(train_df)} points, which is less than the recommended {int(min_required_points)} for 72 hours.")
        except Exception as e:
            print(f"   - Warning: Could not validate training data length due to an error: {e}")

        return train_df, analysis_df

    def scale_features(self, train_df: pd.DataFrame, analysis_df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame]:
        scaler = RobustScaler()
        scaler.fit(train_df[self.feature_columns])
        train_scaled = scaler.transform(train_df[self.feature_columns])
        analysis_scaled = scaler.transform(analysis_df[self.feature_columns])

        train_scaled_df = pd.DataFrame(train_scaled, index=train_df.index, columns=self.feature_columns)
        analysis_scaled_df = pd.DataFrame(analysis_scaled, index=analysis_df.index, columns=self.feature_columns)
        return train_scaled_df, analysis_scaled_df

    def run(self, train_start_time: str, train_end_time: str, analysis_end_time: str) -> Tuple[pd.DataFrame, pd.DataFrame, List[str]]:
        self.load_and_prepare_data()
        self.clean_and_validate_data()
        temp_train_df = self.df.loc[train_start_time:train_end_time]
        self.remove_constant_features(temp_train_df)
        self.remove_correlated_features(temp_train_df)
        train_df, analysis_df = self.split_data(train_start_time, train_end_time, analysis_end_time)
        train_scaled_df, analysis_scaled_df = self.scale_features(train_df, analysis_df)

        return train_scaled_df, analysis_scaled_df, self.feature_columns