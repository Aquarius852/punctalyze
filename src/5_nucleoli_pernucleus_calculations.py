
import os
import numpy as np
import pandas as pd
from scipy import stats
import functools
import importlib.util
import sys
from loguru import logger

# special import, path to script
nucleoli_ana_path = 'src/4_nucleoli_detection.py'

# load the module dynamically due to annoying file name
spec = importlib.util.spec_from_file_location('nucleoli_detection', nucleoli_ana_path)
nucleoli_detection_utils = importlib.util.module_from_spec(spec)
sys.modules['nucleoli_detection_utils'] = nucleoli_detection_utils
spec.loader.exec_module(nucleoli_detection_utils)
aggregate_features_by_group = nucleoli_detection_utils.aggregate_features_by_group

logger.info('import ok')

# configuration
input_folder = 'results/summary_calculations/'
output_folder = 'results/summary_calculations/'


def calculate_nucleus_features(df):
    """Calculate summarized features per nucleus from nucleoli features."""
    
    group_cols = ['image_name', 'nucleus_number']

    # Use pandas groupby + agg with named aggregations
    agg_df = df.groupby(group_cols).agg({
        'nucleoli_minor_axis_length': 'mean',
        'nucleoli_major_axis_length': 'mean',
        'nucleoli_aspect_ratio': 'mean',
        'nucleoli_area': ['mean', 'sum', 'count'],
        'nucleus_size': 'mean',
        'nucleoli_eccentricity': 'mean',
        'nucleoli_cv': 'mean',
        'nucleoli_skew': 'mean',
        'coi2_partition_coeff': 'mean',
        'coi1_partition_coeff': 'mean',
        'nucleus_std': 'mean',
        'nucleus_cv': 'mean',
        'nucleus_skew': 'mean',
        'nucleus_coi1_intensity_mean': 'mean',
        'nucleoli_intensity_mean': 'mean'
    })

    # Flatten MultiIndex columns from aggregation
    agg_df.columns = ['_'.join(col).strip() if isinstance(col, tuple) else col 
        for col in agg_df.columns.values]
    agg_df = agg_df.reset_index()

    # Calculate nucleoli area proportion (%)
    agg_df['nucleoli_area_proportion'] = (agg_df['nucleoli_area_sum'] / agg_df['nucleus_size_mean']) * 100

    # Rename columns for clarity
    agg_df = agg_df.rename(columns={
        'nucleoli_minor_axis_length_mean': 'nucleoli_mean_minor_axis',
        'nucleoli_major_axis_length_mean': 'nucleoli_mean_major_axis',
        'nucleoli_aspect_ratio_mean': 'nucleoli_mean_aspect_ratio',
        'nucleoli_area_mean': 'mean_nucleoli_area',
        'nucleoli_area_count': 'nucleoli_count',
        'nucleoli_eccentricity_mean': 'avg_eccentricity',
        'nucleoli_cv_mean': 'nucleoli_cv_mean',
        'nucleoli_skew_mean': 'nucleoli_skew_mean',
        'coi2_partition_coeff_mean': 'coi2_partition_coeff',
        'coi1_partition_coeff_mean': 'coi1_partition_coeff',
        'nucleus_std_mean': 'nucleus_std',
        'nucleus_cv_mean': 'nucleus_cv',
        'nucleus_skew_mean': 'nucleus_skew',
        'nucleus_coi1_intensity_mean_mean': 'nucleus_coi1_intensity_mean',
        'nucleoli_intensity_mean_mean': 'nucleoli_intensity_mean',
        'nucleus_size_mean': 'nucleus_size'
    })

    return agg_df


def save_nucleus_features(df, features, group_cols=['condition', 'tag', 'rep']):
    # Save raw summary per nucleus
    df.to_csv(f'{output_folder}pernucleus_nucleoli_features.csv', index=False)

    # Average by biological replicate using the aggregate_features_by_group function
    rep_df = aggregate_features_by_group(df, group_cols, features)
    rep_df.to_csv(f'{output_folder}pernucleus_nucleoli_features_reps.csv', index=False)

    # Normalize to nucleus_coi1_intensity_mean
    df_norm = df.copy()
    for col in features:
        df_norm[col] = df_norm[col] / df_norm['nucleus_coi1_intensity_mean']

    df_norm.to_csv(f'{output_folder}pernucleus_nucleoli_features_normalized.csv', index=False)

    # Average normalized data by biological replicate
    rep_norm_df = aggregate_features_by_group(df_norm, group_cols, features)
    rep_norm_df.to_csv(f'{output_folder}pernucleus_nucleoli_features_normalized_reps.csv', index=False)


def save_nucleoli_level_reps(df, features,
                             group_cols=['condition', 'tag', 'rep'],
                             intensity_norm_col='nucleus_coi1_intensity_mean'):

    # --- raw per-nucleolus ---
    df.to_csv(f'{output_folder}nucleoli_features.csv', index=False)

    # --- replicate-averaged (raw) ---
    rep_df = aggregate_features_by_group(df, group_cols, features)
    rep_df.to_csv(f'{output_folder}nucleoli_features_reps.csv', index=False)

    # --- normalized per-nucleolus ---
    df_norm = df.copy()
    for col in features:
        df_norm[col] = df_norm[col] / df_norm[intensity_norm_col]

    df_norm.to_csv(f'{output_folder}nucleoli_features_normalized.csv', index=False)

    # --- replicate-averaged (normalized) ---
    rep_norm_df = aggregate_features_by_group(df_norm, group_cols, features)
    rep_norm_df.to_csv(
        f'{output_folder}nucleoli_features_normalized_reps.csv',
        index=False
    )


if __name__ == '__main__':
    
    # Load feature information
    feature_information = pd.read_csv(f'{input_folder}nucleoli_features.csv')

    # Calculate summarized features per nucleus
    summary = calculate_nucleus_features(feature_information)

    # Add metadata columns
    summary['tag'] = ['EYFP' for name in summary['image_name']]
    summary['condition'] = summary['image_name'].str.split('_').str[1]
    summary['rep'] = summary['image_name'].str.split('_').str[-2]

    # Define features of interest (excluding metadata columns)
    nucleus_features = summary.columns.tolist()
    nucleus_features = [item for item in nucleus_features if '_coords' not in item]
    nucleus_features = ['nucleus_size', 'mean_nucleoli_area', 'nucleoli_area_proportion', 'nucleoli_count',
        'nucleoli_mean_minor_axis', 'nucleoli_mean_major_axis', 'nucleoli_mean_aspect_ratio','avg_eccentricity',
        'nucleoli_cv_mean', 'nucleoli_skew_mean', 'coi2_partition_coeff', 'coi1_partition_coeff', 'nucleus_std',
        'nucleus_cv', 'nucleus_skew', 'nucleus_coi1_intensity_mean', 'nucleus_coi2_intensity_mean', 'nucleoli_intensity_mean']

    # Save dataframes (raw, averaged, normalized, normalized averaged)
    save_nucleus_features(summary, nucleus_features)

    # Ensure metadata exists
    feature_information['tag'] = 'FBL'
    feature_information['condition'] = feature_information['image_name'].str.split('_').str[1]
    feature_information['rep'] = feature_information['image_name'].str.split('_').str[-2]

    # Define nucleoli-level features (MUST match plotting script)
    nucleoli_features = [
        'nucleoli_area',
        'nucleoli_eccentricity',
        'nucleoli_aspect_ratio',
        'nucleoli_circularity',
        'nucleoli_cv',
        'nucleoli_skew',
        'coi2_partition_coeff',
        'coi1_partition_coeff',
        'nucleus_std',
        'nucleus_cv',
        'nucleus_skew',
        'nucleoli_intensity_mean',
        'nucleoli_intensity_mean_in_coi2',
        'nucleoli_enrichment_coi1',
        'nucleoli_mass_coi1'
    ]
    
    # Generate all nucleoli-level summary files
    save_nucleoli_level_reps(feature_information, nucleoli_features)

    logger.info('saved nucleoli feature averaged-per-nucleus dataframes')