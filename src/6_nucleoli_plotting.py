import os
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from itertools import combinations
from statannotations.Annotator import Annotator
from loguru import logger
from scipy import stats

logger.info('import ok')

# plotting config
plt.rcParams.update({'font.size': 14})
sns.set_palette('Paired')

# configuration
input_folder = 'results/summary_calculations/'
output_folder = 'results/plotting/'

os.makedirs(output_folder, exist_ok=True)


def load_summary_data(input_folder):
    return {
        'nucleoli_features': pd.read_csv(f'{input_folder}nucleoli_features.csv'),
        'nucleoli_features_reps': pd.read_csv(f'{input_folder}nucleoli_features_reps.csv'),
        'nucleoli_features_normalized': pd.read_csv(f'{input_folder}nucleoli_features_normalized.csv'),
        'nucleoli_features_normalized_reps': pd.read_csv(f'{input_folder}nucleoli_features_normalized_reps.csv'),
        'pernucleus': pd.read_csv(f'{input_folder}pernucleus_nucleoli_features.csv'),
        'pernucleus_reps': pd.read_csv(f'{input_folder}pernucleus_nucleoli_features_reps.csv')
    }


# --- Plotting Functions ---
def plot_stats(data_raw, data_agg, features, title, save_name, x='condition', hue='tag', pairs=None, order=None, replicate_col='rep', nucleus_id_col='image_name'):
    # --- compute N (replicates) ---
    if replicate_col in data_agg.columns:
        N_per_group = data_agg.groupby([x])[replicate_col].nunique()
    else:
        # fallback: assume each row in aggregated data is a replicate
        N_per_group = data_agg.groupby([x]).size()

    # --- compute n (nuclei) ---
    if hue is None:
        if {'image_name', 'nucleus_number'}.issubset(data_raw.columns):
            n_per_group = (
                data_raw
                .drop_duplicates(subset=['image_name', 'nucleus_number'])
                .groupby(x)
                .size()
            )
        else:
            # fallback if nucleus_number not available
            n_per_group = data_raw.groupby(x).size()

    else:
        if nucleus_id_col in data_raw.columns:
            n_per_group = data_raw.groupby([x, hue])[nucleus_id_col].nunique()
        else:
            n_per_group = data_raw.groupby([x, hue]).size()


    # --- format text ---
    summary_lines = []
    for cond in N_per_group.index:
        N_val = N_per_group.get(cond, np.nan)
        n_val = n_per_group.get(cond, np.nan)
        summary_lines.append(f"{cond}: N={N_val}, n={n_val}")
    N_min, N_max = N_per_group.min(), N_per_group.max()
    n_min, n_max = n_per_group.min(), n_per_group.max()
    summary_text = f"N = {N_min}–{N_max} images\nn = {n_min}–{n_max} nucleuss"
    
    fig, axes = plt.subplots(nrows=6, ncols=3, figsize=(20, 30))
    axes = axes.flatten()

    if hue is None:
        for i, feature in enumerate(features):
            ax = axes[i]
            sns.stripplot(data=data_raw, x=x, y=feature, dodge=True, edgecolor='white',
                        linewidth=1, size=8, alpha=0.1, order=order, ax=ax, zorder=0)
            sns.violinplot(data=data_agg, x=x, y=feature, order=order, color='gray', ax=ax, zorder=1)
            sns.stripplot(data=data_agg, x=x, y=feature, dodge=True, edgecolor='k',
                        linewidth=1, size=8, order=order, ax=ax, zorder=2)
            sns.despine()

            if pairs:
                annotator = Annotator(ax, pairs, data=data_agg, x=x, y=feature, order=order)
                annotator.configure(test='t-test_ind', verbose=0)
                annotator.apply_test()
                annotator.annotate()
        
        for ax in axes[len(features):]:
            ax.axis('off')

        fig.text(1.0, 0.9, summary_text)
        fig.suptitle(title, fontsize=18, y=0.99)
        fig.tight_layout()
        fig.savefig(os.path.join(output_folder, save_name), bbox_inches='tight', pad_inches=0.1, dpi=300)
        plt.close(fig)


    else:   
        for i, feature in enumerate(features):
            ax = axes[i]
            sns.stripplot(data=data_raw, x=x, y=feature, dodge=True, edgecolor='white',
                        linewidth=1, size=8, alpha=0.4, hue=hue, order=order, ax=ax)
            sns.stripplot(data=data_agg, x=x, y=feature, dodge=True, edgecolor='k',
                        linewidth=1, size=8, hue=hue, order=order, ax=ax)
            sns.boxplot(data=data_agg, x=x, y=feature, palette=['.9'], hue=hue,
                        order=order, ax=ax)

            ax.legend_.remove()
            sns.despine()

            if pairs:
                annotator = Annotator(ax, pairs, data=data_agg, x=x, y=feature, hue=hue, order=order)
                annotator.configure(test='Mann-Whitney', verbose=0)
                annotator.apply_test()
                annotator.annotate()

        for ax in axes[len(features):]:
            ax.axis('off')

        fig.text(1.0, 0.9, summary_text)
        fig.suptitle(title, fontsize=18, y=0.99)
        handles, labels = ax.get_legend_handles_labels()
        fig.tight_layout()
        fig.legend(handles, labels, bbox_to_anchor=(1.1, 1), title=hue)
        fig.savefig(os.path.join(output_folder, save_name), bbox_inches='tight', pad_inches=0.1, dpi=300)
        plt.close(fig)

def calculate_pairwise_stats(data, feature, condition1, condition2):
    """
    Calculate statistical comparison between two conditions for a specific feature.
    Uses only the mean values from replicates (3 points per condition).
    
    Parameters:
    -----------
    data : pd.DataFrame
        Aggregated data with replicate means
    feature : str
        Column name of the feature to analyze
    condition1, condition2 : str
        Names of conditions to compare
    
    Returns:
    --------
    dict : Statistics including mean, std, t-test p-value, and effect size
    """
    group1 = data[data['condition'] == condition1][feature].dropna().values
    group2 = data[data['condition'] == condition2][feature].dropna().values
    
    if len(group1) < 2 or len(group2) < 2:
        return {'n1': len(group1), 'n2': len(group2), 'p_value': np.nan, 'effect_size': np.nan}
    
    mean1, mean2 = group1.mean(), group2.mean()
    std1, std2 = group1.std(), group2.std()
    
    # Independent t-test
    t_stat, p_value = stats.ttest_ind(group1, group2)
    
    # Calculate Cohen's d effect size
    pooled_std = np.sqrt(((len(group1)-1)*std1**2 + (len(group2)-1)*std2**2) / (len(group1) + len(group2) - 2))
    cohens_d = (mean1 - mean2) / pooled_std if pooled_std > 0 else 0
    
    return {
        'n1': len(group1), 'n2': len(group2),
        'mean1': mean1, 'mean2': mean2,
        'std1': std1, 'std2': std2,
        't_stat': t_stat, 'p_value': p_value,
        'effect_size': cohens_d
    }


def plot_pairwise_comparison(data, features, condition1, condition2, title, output_file):
    """
    Create focused plot comparing two conditions across selected features.
    Shows individual replicate means as scatter points.
    
    Parameters:
    -----------
    data : pd.DataFrame
        Aggregated data with replicate means
    features : list
        List of feature names to plot
    condition1, condition2 : str
        Conditions to compare
    title : str
        Plot title
    output_file : str
        Output filename
    
    Returns:
    --------
    dict : Statistics results {feature: {stats_dict}}
    """
    # Filter data for the two conditions
    plot_data = data[data['condition'].isin([condition1, condition2])].copy()
    
    if plot_data.empty:
        logger.warning(f"No data found for conditions {condition1}, {condition2}")
        return {}
    
    # Create subplots
    n_features = len(features)
    n_cols = 2
    n_rows = (n_features + n_cols - 1) // n_cols
    
    fig, axes = plt.subplots(nrows=n_rows, ncols=n_cols, figsize=(12, 5*n_rows))
    if n_rows == 1:
        axes = axes.reshape(1, -1)
    axes = axes.flatten()
    
    # Statistical results storage
    stats_results = {}
    
    for i, feature in enumerate(features):
        ax = axes[i]
        
        # Plot data
        for condition in [condition1, condition2]:
            cond_data = plot_data[plot_data['condition'] == condition][feature].dropna()
            x_pos = [condition] * len(cond_data)
            # Add jitter
            x_jitter = np.random.normal([condition1, condition2].index(condition), 0.04, len(cond_data))
            ax.scatter(x_jitter, cond_data, s=150, alpha=0.6, edgecolors='k', linewidth=1.5)
        
        # Calculate statistics
        stats_dict = calculate_pairwise_stats(plot_data, feature, condition1, condition2)
        stats_results[feature] = stats_dict
        
        # Add mean lines
        for j, condition in enumerate([condition1, condition2]):
            cond_data = plot_data[plot_data['condition'] == condition][feature].dropna()
            mean_val = cond_data.mean()
            ax.hlines(mean_val, j-0.2, j+0.2, colors='red', linewidth=2, label='Mean' if j == 0 else '')
        
        # Formatting
        ax.set_xticks([0, 1])
        ax.set_xticklabels([condition1, condition2])
        ax.set_ylabel(feature, fontsize=12, fontweight='bold')
        ax.grid(axis='y', alpha=0.3)
        sns.despine()
        
        # Add statistics text (will be updated with correction later)
        p_val = stats_dict.get('p_value', np.nan)
        effect_size = stats_dict.get('effect_size', np.nan)
        sig_marker = '**' if p_val < 0.01 else '*' if p_val < 0.05 else 'ns'
        
        text_str = f"p = {p_val:.4f} {sig_marker}\nCohen's d = {effect_size:.3f}"
        ax.text(0.5, 0.95, text_str, transform=ax.transAxes, fontsize=10,
                verticalalignment='top', horizontalalignment='center',
                bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
    
    # Hide unused subplots
    for ax in axes[n_features:]:
        ax.axis('off')
    
    fig.suptitle(title, fontsize=16, fontweight='bold', y=0.995)
    fig.tight_layout()
    fig.savefig(os.path.join(output_folder, output_file), bbox_inches='tight', dpi=300)
    plt.close(fig)
    
    return stats_results


def apply_bonferroni_and_plot(all_results, data, features, comparison_pairs, output_folder):
    """
    Apply Bonferroni correction per feature across all comparisons and generate corrected plots.
    
    Parameters:
    -----------
    all_results : dict
        Results from all comparisons {(cond1, cond2): {feature: stats_dict}}
    data : pd.DataFrame
        Aggregated data with replicate means
    features : list
        List of feature names
    comparison_pairs : list
        List of (condition1, condition2) tuples
    output_folder : str
        Output directory for plots
    """
    # Bonferroni correction: for each feature, multiply p-values by number of comparisons
    n_comparisons = len(comparison_pairs)
    
    # Store corrected results
    corrected_results = {}
    
    # Apply Bonferroni correction per feature
    for feature in features:
        corrected_results[feature] = {}
        for pair in comparison_pairs:
            if pair in all_results and feature in all_results[pair]:
                p_val = all_results[pair][feature].get('p_value', np.nan)
                p_corrected = min(p_val * n_comparisons, 1.0)  # Cap at 1.0
                
                corrected_results[feature][pair] = {
                    'p_original': p_val,
                    'p_corrected': p_corrected
                }
    
    # Regenerate plots with corrected p-values
    for (cond1, cond2) in comparison_pairs:
        plot_data = data[data['condition'].isin([cond1, cond2])].copy()
        
        if plot_data.empty:
            continue
        
        n_features_plot = len(features)
        n_cols = 2
        n_rows = (n_features_plot + n_cols - 1) // n_cols
        
        fig, axes = plt.subplots(nrows=n_rows, ncols=n_cols, figsize=(12, 5*n_rows))
        if n_rows == 1:
            axes = axes.reshape(1, -1)
        axes = axes.flatten()
        
        for i, feature in enumerate(features):
            ax = axes[i]
            
            # Plot data
            for condition in [cond1, cond2]:
                cond_data = plot_data[plot_data['condition'] == condition][feature].dropna()
                x_jitter = np.random.normal([cond1, cond2].index(condition), 0.04, len(cond_data))
                ax.scatter(x_jitter, cond_data, s=150, alpha=0.6, edgecolors='k', linewidth=1.5)
            
            # Add mean lines
            for j, condition in enumerate([cond1, cond2]):
                cond_data = plot_data[plot_data['condition'] == condition][feature].dropna()
                mean_val = cond_data.mean()
                ax.hlines(mean_val, j-0.2, j+0.2, colors='red', linewidth=2)
            
            # Formatting
            ax.set_xticks([0, 1])
            ax.set_xticklabels([cond1, cond2])
            ax.set_ylabel(feature, fontsize=12, fontweight='bold')
            ax.grid(axis='y', alpha=0.3)
            sns.despine()
            
            # Get statistics with corrected p-value
            if (cond1, cond2) in all_results and feature in all_results[(cond1, cond2)]:
                stats_dict = all_results[(cond1, cond2)][feature]
                p_corrected = corrected_results[feature][(cond1, cond2)]['p_corrected']
                effect_size = stats_dict.get('effect_size', np.nan)
                
                # Significance marker based on CORRECTED p-value
                sig_marker = '**' if p_corrected < 0.01 else '*' if p_corrected < 0.05 else 'ns'
                
                p_orig = stats_dict.get('p_value', np.nan)
                text_str = f"p = {p_orig:.4f} → {p_corrected:.4f} {sig_marker}\nCohen's d = {effect_size:.3f}"
                ax.text(0.5, 0.95, text_str, transform=ax.transAxes, fontsize=9,
                        verticalalignment='top', horizontalalignment='center',
                        bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
        
        # Hide unused subplots
        for ax in axes[n_features_plot:]:
            ax.axis('off')
        
        title = f'Nuclear Enrichment & Partition Coefficient (Bonferroni Corrected):\n{cond1} vs {cond2}'
        fig.suptitle(title, fontsize=16, fontweight='bold', y=0.995)
        fig.tight_layout()
        fig.savefig(os.path.join(output_folder, f'pairwise_corrected_{cond1}_vs_{cond2}.png'), 
                    bbox_inches='tight', dpi=300)
        plt.close(fig)
    
    # Log corrected statistics
    logger.info(f"\n{'='*80}")
    logger.info(f"BONFERRONI-CORRECTED STATISTICS (corrected for {n_comparisons} comparisons per feature)")
    logger.info(f"{'='*80}")
    for feature in features:
        logger.info(f"\n{feature}:")
        for (cond1, cond2) in comparison_pairs:
            if feature in corrected_results and (cond1, cond2) in corrected_results[feature]:
                p_orig = corrected_results[feature][(cond1, cond2)]['p_original']
                p_corr = corrected_results[feature][(cond1, cond2)]['p_corrected']
                sig = '**' if p_corr < 0.01 else '*' if p_corr < 0.05 else 'ns'
                logger.info(f"  {cond1} vs {cond2}: p = {p_orig:.4f} → {p_corr:.4f} {sig}")



if __name__ == '__main__':
    logger.info('Loading data...')
    dfs = load_summary_data(input_folder)

    nucleoli_features_raw = ['nucleoli_area', 'nucleoli_eccentricity', 'nucleoli_aspect_ratio',
                'nucleoli_circularity', 'nucleoli_cv', 'nucleoli_skew',
                'nucleus_std',
                'nucleus_cv', 'nucleus_skew', 'coi1_nucleoli_intensity', 'coi1_nucleoli_mean_intensity',
                'coi2_nucleoli_intensity', 'coi2_nucleoli_mean_intensity']

    nucleoli_features_normalized = nucleoli_features_raw + ['coi1_nucleolar_enrichment', 'coi2_nucleolar_enrichment',
                'coi1_partition_coefficient', 'coi2_partition_coefficient']

    pernucleus_features = ['nucleus_size', 'mean_nucleoli_area', 'nucleoli_area_proportion', 'nucleoli_count',
            'nucleoli_mean_minor_axis', 'nucleoli_mean_major_axis', 'nucleoli_mean_aspect_ratio','avg_eccentricity',
            'nucleoli_cv_mean', 'nucleoli_skew_mean', 'nucleus_std',
            'nucleus_cv', 'nucleus_skew', 'nucleus_coi1_intensity_mean', 'nucleus_coi2_intensity_mean']

    # could use combinations function to generate pairs dynamically, but here we define them explicitly
    conditions = dfs['nucleoli_features']['condition'].unique().tolist()
    paired_conditions = combinations(conditions, 2)
    paired_list = list(paired_conditions)
    paired_list = [pair for pair in paired_list if 'WT' in pair]  # only compare to WT
    order = ["DFMO","SP1HR","SP2HR","SP4HR","SP24HR"]
    # palette = ['#A6CEE3', '#1F78B4', '#F5CB5C']
    palette = sns.color_palette('tab10', n_colors=len(conditions))

    # prepare plotting configuration as [(title, features, raw_df, reps_df), (etc...)]
    plotting_configs = [
        ('per nucleoli, raw', nucleoli_features_raw, dfs['nucleoli_features'], dfs['nucleoli_features_reps'], 'pernucleoli_raw.png'),
        ('per nucleoli, normalized', nucleoli_features_normalized, dfs['nucleoli_features_normalized'], dfs['nucleoli_features_normalized_reps'], 'pernucleoli_normalized.png'),
        ('per nucleus, raw', pernucleus_features, dfs['pernucleus'], dfs['pernucleus_reps'], 'pernucleus_raw.png'),
    ]

    # TODO make plotting more dynamic to handle stats/no-stats cases
    logger.info('Generating paired plots with stats...')
    for title, features, raw_df, reps_df, filename in plotting_configs:
        title
        plot_stats(raw_df, reps_df, features, f'Calculated Parameters - {title}', filename,
                   x='condition', hue=None, pairs=paired_list, order=order)
    
    # --- Focused pairwise comparison: nuclear enrichment and partition coefficient ---
    logger.info('\nGenerating focused pairwise comparison plots...')
    
    enrichment_partition_features = ['coi1_nucleolar_enrichment', 'coi2_nucleolar_enrichment',
                                     'coi1_partition_coefficient', 'coi2_partition_coefficient']
    
    # Available conditions to compare
    available_conditions = dfs['nucleoli_features_normalized_reps']['condition'].unique()
    
    # Define comparison pairs
    comparison_pairs = [
        ('DFMO', 'SP1HR'),
        ('DFMO', 'SP24HR')
    ]
    
    # Store results from all comparisons
    all_comparison_results = {}
    
    for cond1, cond2 in comparison_pairs:
        if cond1 in available_conditions and cond2 in available_conditions:
            results = plot_pairwise_comparison(
                dfs['nucleoli_features_normalized_reps'],
                enrichment_partition_features,
                cond1,
                cond2,
                f'Nuclear Enrichment & Partition Coefficient:\n{cond1} vs {cond2}',
                f'pairwise_{cond1}_vs_{cond2}.png'
            )
            all_comparison_results[(cond1, cond2)] = results
            logger.info(f"✓ Pairwise comparison plot saved: {cond1} vs {cond2}")
        else:
            logger.warning(f"Condition(s) not found. Looking for '{cond1}' and '{cond2}' in: {available_conditions.tolist()}")
    
    # Apply Bonferroni correction and generate corrected plots
    if all_comparison_results:
        apply_bonferroni_and_plot(
            all_comparison_results,
            dfs['nucleoli_features_normalized_reps'],
            enrichment_partition_features,
            comparison_pairs,
            output_folder
        )
        logger.info(f"✓ Bonferroni-corrected plots generated")
