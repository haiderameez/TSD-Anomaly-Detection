import pandas as pd
import numpy as np
import os
from typing import List

from src.data_preprocessor import DataPreprocessor
from src.unified_detector import UnifiedAnomalyDetector

INPUT_CSV_PATH = 'data/data.csv'
PREPROCESSED_DIR = 'preprocessed_data'
OUTPUT_DIR = 'final_output'
FINAL_OUTPUT_CSV_PATH = os.path.join(OUTPUT_DIR, 'final_output_with_anomalies.csv')

#set time ranges according to the new data that you want (structure should be the same)
TRAIN_START_TIME = '2004-01-01 00:00:00'
TRAIN_END_TIME = '2004-01-05 23:59:00'
ANALYSIS_END_TIME = '2004-01-19 07:59:00'


def run_preprocessing(input_path: str, output_dir: str) -> (pd.DataFrame, pd.DataFrame, List[str]):
    print("Data Preprocessing")
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    preprocessor = DataPreprocessor(file_path=input_path)
    train_scaled, analysis_scaled, feature_names = preprocessor.run(
        TRAIN_START_TIME, TRAIN_END_TIME, ANALYSIS_END_TIME
    )
    
    train_scaled.to_csv(os.path.join(output_dir, 'train_scaled.csv'))
    analysis_scaled.to_csv(os.path.join(output_dir, 'analysis_scaled.csv'))
    
    print("Preprocessing Finished\n")
    return train_scaled, analysis_scaled, feature_names


def run_analysis(train_df: pd.DataFrame, analysis_df: pd.DataFrame, features: List[str]) -> pd.DataFrame:
    
    vae_latent_dim = max(2, min(8, len(features) // 5))
    
    detector = UnifiedAnomalyDetector(
        lof_neighbors=40,
        mcd_support=0.8,
        vae_latent_dim=vae_latent_dim,
        vae_epochs=30
    )
    
    detector.train(train_df, features)
    results = detector.predict(analysis_df)

    return results


def validate_and_summarize(final_df: pd.DataFrame):
    print("Prediction and Summary")
    
    train_period_scores = final_df.loc[TRAIN_START_TIME:TRAIN_END_TIME]['Abnormality_score']
    mean_train_score = train_period_scores.mean()
    max_train_score = train_period_scores.max()
    
    print("Validation Results (Training Period):")
    print(f"  - Mean Abnormality Score: {mean_train_score:.2f} (Success: < 10)")
    print(f"  - Max Abnormality Score: {max_train_score:.2f} (Success: < 25)")
    
    if mean_train_score >= 10 or max_train_score >= 25:
        print("  - WARNING: Validation criteria not met. Model may be overly sensitive.")
    else:
        print("  - SUCCESS: Model passes validation criteria.")

    print("\nAnomaly Summary:")
    scores = final_df['Abnormality_score'].values
    print(f"  - Normal (0-10):      {np.sum((scores >= 0) & (scores <= 10))}")
    print(f"  - Slight (11-30):     {np.sum((scores > 10) & (scores <= 30))}")
    print(f"  - Moderate (31-60):   {np.sum((scores > 30) & (scores <= 60))}")
    print(f"  - Significant (61-90):{np.sum((scores > 60) & (scores <= 90))}")
    print(f"  - Severe (91-100):    {np.sum(scores > 90)}")


def main():
    train_scaled_df, analysis_scaled_df, feature_names = run_preprocessing(
        INPUT_CSV_PATH, PREPROCESSED_DIR
    )
    
    results_df = run_analysis(train_scaled_df, analysis_scaled_df, feature_names)
    
    original_df = pd.read_csv(INPUT_CSV_PATH, parse_dates=['Time'], index_col='Time')
    
    final_output_df = original_df.loc[analysis_scaled_df.index].copy()
    final_output_df = final_output_df.join(results_df)

    if not os.path.exists(OUTPUT_DIR):
        os.makedirs(OUTPUT_DIR)
    final_output_df.to_csv(FINAL_OUTPUT_CSV_PATH)
    print(f"Final output saved to: {FINAL_OUTPUT_CSV_PATH}\n")
    
    validate_and_summarize(final_output_df)


if __name__ == '__main__':
    try:
        main()
    except Exception as e:
        print(f"\nAn unhandled error occurred: {e}")
        raise
