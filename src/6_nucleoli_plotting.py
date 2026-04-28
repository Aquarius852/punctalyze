import os
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from itertools import combinations
from statannotations.Annotator import Annotator
from loguru import logger

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
        'pernucleus_reps': pd.read_csv(f'{input_folder}pernucleus_nucleoli_features_reps.csv'),
        'pernucleus_norm': pd.read_csv(f'{input_folder}pernucleus_nucleoli_features_normalized.csv'),
        'pernucleus_norm_reps': pd.read_csv(f'{input_folder}pernucleus_nucleoli_features_normalized_reps.csv')
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
        if nucleus_id_col in data_raw.columns:
            n_per_group = data_raw.groupby(x)[nucleus_id_col].nunique()
        else:
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


def plot_partition_coefficients(data_raw, data_agg, save_name, x='tag', hue='condition', order=None):
    palette = ['#A6CEE3', '#1F78B4', '#F5CB5C']

    raw = pd.melt(data_raw, id_vars=['image_name', 'tag', 'condition'],
                  value_vars=['coi1_partition_coeff', 'coi2_partition_coeff'],
                  var_name='channel', value_name='partition_coeff')

    agg = pd.melt(data_agg, id_vars=['rep', 'tag', 'condition'],
                  value_vars=['coi1_partition_coeff', 'coi2_partition_coeff'],
                  var_name='channel', value_name='partition_coeff')

    g = sns.FacetGrid(agg, col='channel', height=4.5, aspect=0.8)
    g.map_dataframe(sns.boxplot, x=x, y='partition_coeff', palette=['.9'], hue=hue, hue_order=order, zorder=0)
    g.map_dataframe(sns.stripplot, x=x, y='partition_coeff', dodge=True, edgecolor='k',
                    linewidth=1, hue=hue, palette=palette, hue_order=order, zorder=2, size=8)

    for ax_i, category in enumerate(g.col_names):
        ax = g.axes.flat[ax_i]
        subset = raw[raw['channel'] == category]
        sns.stripplot(data=subset, x=x, y='partition_coeff', dodge=True,
                      edgecolor='white', linewidth=1, alpha=0.4, hue=hue,
                      palette=palette, hue_order=order, zorder=1, size=8, ax=ax)
        ax.get_legend().remove()
        ax.set_xticklabels(['COI1', 'COI2'])
        ax.set_xlabel('')

    g.set_titles(col_template='{col_name}')
    g.tight_layout()
    g.fig.savefig(os.path.join(output_folder, save_name), bbox_inches='tight', pad_inches=0.1, dpi=300)
    plt.close(g.fig)


if __name__ == '__main__':
    logger.info('Loading data...')
    dfs = load_summary_data(input_folder)

    nucleoli_features = ['nucleoli_area', 'nucleoli_eccentricity', 'nucleoli_aspect_ratio',
                'nucleoli_circularity', 'nucleoli_cv', 'nucleoli_skew',
                'coi2_partition_coeff', 'coi1_partition_coeff', 'nucleus_std',
                'nucleus_cv', 'nucleus_skew']

    pernucleus_features = ['nucleus_size', 'mean_nucleoli_area', 'nucleoli_area_proportion', 'nucleoli_count',
            'nucleoli_mean_minor_axis', 'nucleoli_mean_major_axis', 'nucleoli_mean_aspect_ratio','avg_eccentricity',
            'nucleoli_cv_mean', 'nucleoli_skew_mean', 'coi2_partition_coeff', 'coi1_partition_coeff', 'nucleus_std',
            'nucleus_cv', 'nucleus_skew', 'nucleus_coi1_intensity_mean']

    # could use combinations function to generate pairs dynamically, but here we define them explicitly
    conditions = dfs['nucleoli_features']['condition'].unique().tolist()
    paired_conditions = combinations(conditions, 2)
    paired_list = list(paired_conditions)
    paired_list = [pair for pair in paired_list if 'WT' in pair]  # only compare to WT
    order = sorted(conditions)
    # palette = ['#A6CEE3', '#1F78B4', '#F5CB5C']
    palette = sns.color_palette('tab10', n_colors=len(conditions))

    # prepare plotting configuration as [(title, features, raw_df, reps_df), (etc...)]
    plotting_configs = [
        ('per nucleoli, raw', nucleoli_features, dfs['nucleoli_features'], dfs['nucleoli_features_reps'], 'pernucleoli_raw.png'),
        ('per nucleoli, normalized', nucleoli_features, dfs['nucleoli_features_normalized'], dfs['nucleoli_features_normalized_reps'], 'pernucleoli_normalized.png'),
        ('per nucleus, raw', pernucleus_features, dfs['pernucleus'], dfs['pernucleus_reps'], 'pernucleus_raw.png'),
        ('per nucleus, normalized', pernucleus_features, dfs['pernucleus_norm'], dfs['pernucleus_norm_reps'], 'pernucleus_normalized.png'),
    ]

    # TODO make plotting more dynamic to handle stats/no-stats cases
    logger.info('Generating paired plots with stats...')
    for title, features, raw_df, reps_df, filename in plotting_configs:
        title
        plot_stats(raw_df, reps_df, features, f'Calculated Parameters - {title}', filename,
                   x='condition', hue=None, pairs=paired_list, order=order)

    # TODO fix partition coefficient plots
    logger.info('Generating partition coefficient plots...')
    plot_partition_coefficients(dfs['pernucleus'], dfs['pernucleus_reps'], 'condition-paired_pernucleus_raw_partition-only.png', order=order)
