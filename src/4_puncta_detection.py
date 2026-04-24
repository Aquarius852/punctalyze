"""
Detect and analyze features of nucleoli per nucleus
(Modified from your puncta pipeline)

Updated so that:
- nucleoli segmentation can use one channel
- nucleolar/nuclear intensity measurements can use another channel
- morphology always comes from the segmented nucleolar mask
"""

import os
import importlib.util
import sys
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from matplotlib_scalebar.scalebar import ScaleBar
import skimage.io
from skimage import measure, segmentation, morphology
from skimage.filters import gaussian, threshold_otsu
from skimage.feature import peak_local_max
from scipy import stats
from scipy.stats import skewtest
from scipy import ndimage as ndi
from loguru import logger
import functools

print(os.getcwd())
folder = 'results/cellpose_masking/'

BASE_DIR = r"C:\Users\kylee\Lab Documents\Fazal Lab\Punctalyze\punctalyze"

# special import, path to script
napari_utils_path = os.path.join(BASE_DIR, 'src', '3_napari.py')  # adjust as needed

# load the module dynamically due to annoying file name
spec = importlib.util.spec_from_file_location("napari", napari_utils_path)
napari_utils = importlib.util.module_from_spec(spec)
sys.modules["napari_utils"] = napari_utils
spec.loader.exec_module(napari_utils)
remove_saturated_cells = napari_utils.remove_saturated_cells

logger.info('import ok')

# plotting setup
plt.rcParams.update({'font.size': 14})
sns.set_palette('Paired')

# --- configuration ---
SAT_FRAC_CUTOFF = 0.01  # for consistency with remove_saturated_cells

# ----------------------------
# CHANNEL CONFIGURATION
# ----------------------------
# SEGMENTATION_CHANNEL:
#   used to detect nucleoli / create nucleolar masks
#
# PRIMARY_MEASURE_CHANNEL:
#   used for nucleoli_intensity_mean, nucleoli_cv, nucleoli_skew,
#   nucleus_coi1_intensity_mean, nucleus_cv, nucleus_skew
#
# SECONDARY_MEASURE_CHANNEL:
#   optional comparison channel measured inside the SAME masks

SEGMENTATION_CHANNEL = 0
PRIMARY_MEASURE_CHANNEL = 1
SECONDARY_MEASURE_CHANNEL = 1

SEGMENTATION_CHANNEL_NAME = 'fbl'
PRIMARY_MEASURE_CHANNEL_NAME = 's9'
SECONDARY_MEASURE_CHANNEL_NAME = 's9'

SCALE_PX = (294.67 / 2720)  # size of one pixel in units specified by the next constant
SHOW_SURVIVORS_ONLY = True  # proofs show nucleoli masks after filtering

# --- quantification region ---
# nucleoli detection is typically done inside nuclei

QUANT_REGION = "nucleus"

# --- nucleoli detection parameters ---
NUC_SMOOTH_SIGMA = 2.0        # stronger smoothing to suppress texture
THR_METHOD = "otsu"           # "otsu" or "mad"
THR_K = 2.5                   # only used if THR_METHOD="mad"
MIN_NUCLEOLI_SIZE = 250
MAX_NUCLEOLI_SIZE = 8000
FILL_HOLES = True

# morphology for joining fragmented nucleolar regions
CLOSING_RADIUS = 2
OPENING_RADIUS = 0

# keep off for now
SPLIT_TOUCHING = False
PEAK_MIN_DIST = 12
WATERSHED_COMPACTNESS = 0.0

SCALE_UNIT = 'um'

image_folder  = os.path.join(BASE_DIR, "results")
output_folder   = os.path.join(BASE_DIR, "results", "summary_calculations")
mask_folder = os.path.join(BASE_DIR, "results", "napari_masking")
proofs_folder = os.path.join(BASE_DIR, "results", "proofs")

for folder in [output_folder, proofs_folder]:
    if not os.path.exists(folder):
        os.mkdir(folder)


def feature_extractor(mask, properties=None):
    """
    Extract richer morphology + intensity-ready geometry from a labeled mask.
    NOTE: 'coords' is kept for proofing (you drop it later).
    """
    if properties is None:
        properties = [
            'label', 'area',
            'eccentricity',
            'major_axis_length', 'minor_axis_length',
            'orientation',
            'perimeter', 'perimeter_crofton',
            'solidity', 'convex_area',
            'bbox',
            'moments_hu',
            'coords'
        ]

    props = measure.regionprops_table(mask, properties=properties)
    df = pd.DataFrame(props)

    # regionprops_table expands bbox into bbox-0..bbox-3
    bbox_cols = [c for c in df.columns if c.startswith("bbox-")]
    if len(bbox_cols) == 4:
        minr, minc, maxr, maxc = (df["bbox-0"], df["bbox-1"], df["bbox-2"], df["bbox-3"])
        bbox_area = (maxr - minr) * (maxc - minc)
        df["bbox_area"] = bbox_area.astype(float)
        df["extent"] = df["area"] / (df["bbox_area"] + 1e-9)

    return df


def load_images(image_folder):
    images = {}
    for fn in os.listdir(image_folder):
        if fn.endswith('.npy'):
            name = fn.removesuffix('.npy')
            images[name] = np.load(f'{image_folder}/{fn}')
    return images


def load_masks(mask_folder):
    masks = {}
    for fn in os.listdir(mask_folder):
        if fn.endswith('_mask.npy'):
            name = fn.removesuffix('_mask.npy')
            masks[name] = np.load(f'{mask_folder}/{fn}', allow_pickle=True)
    return masks


def build_quant_masks(masks, region="cell"):
    """
    Returns label masks defining the region in which detection happens.
    For "nucleus", returns labeled nuclei.
    """
    quant_masks = {}

    for name, m in masks.items():
        cell_mask, nuc_mask = m[0], m[1]

        if region == "cell":
            quant_masks[name] = cell_mask

        elif region == "nucleus":
            quant_masks[name] = morphology.label(nuc_mask > 0)

        else:
            raise ValueError(f"Unknown QUANT_REGION: {region}")

    return quant_masks


def generate_cytoplasm_masks(masks):
    logger.info('removing nuclei from cell masks...')
    cyto_masks = {}
    for name, img in masks.items():
        cell_mask, nuc_mask = img[0], img[1]
        cell_bin = (cell_mask > 0).astype(int)
        nuc_bin = (nuc_mask > 0).astype(int)

        single_cyto = []
        labels = np.unique(cell_mask)
        if labels.size > 1:
            for lbl in labels[labels != 0]:
                cyto = np.where(cell_mask == lbl, cell_bin, 0)
                cyto_minus_nuc = cyto & ~nuc_bin
                if np.any(cyto_minus_nuc):
                    single_cyto.append(np.where(cyto_minus_nuc, lbl, 0))
                else:
                    single_cyto.append(np.zeros_like(cell_mask, dtype=int))
        else:
            single_cyto.append(np.zeros_like(cell_mask, dtype=int))

        cyto_masks[name] = sum(single_cyto)
    logger.info('cytoplasm masks created.')
    return cyto_masks


def filter_saturated_images(images, quant_masks, masks):
    """
    Return a dict per image with explicit channel roles:
    - segmentation image
    - primary measurement image
    - secondary measurement image
    - filtered nucleus labels
    """
    logger.info('filtering saturated cells...')
    filtered = {}
    for name, img in images.items():
        stack = np.stack([
            img[SECONDARY_MEASURE_CHANNEL],
            img[PRIMARY_MEASURE_CHANNEL],
            quant_masks[name]
        ])

        cells_or_nuclei_filtered = remove_saturated_cells(
            image_stack=stack,
            mask_stack=masks[name],
            COI=1  # PRIMARY_MEASURE_CHANNEL is stack index 1 here
        )

        filtered[name] = {
            "secondary": img[SECONDARY_MEASURE_CHANNEL],
            "primary": img[PRIMARY_MEASURE_CHANNEL],
            "segmentation": img[SEGMENTATION_CHANNEL],
            "mask": cells_or_nuclei_filtered
        }

    logger.info('saturated cells filtered.')
    return filtered


# ----------------------------
# NUCLEOLI DETECTION FUNCTIONS
# ----------------------------
def _mad(x):
    """Median absolute deviation (robust std proxy)."""
    x = np.asarray(x)
    med = np.median(x)
    return np.median(np.abs(x - med)) + 1e-9


def detect_nucleoli_labels_for_image(
    nuc_img,
    nucleus_labels,
    *,
    smooth_sigma=NUC_SMOOTH_SIGMA,
    thr_method=THR_METHOD,
    thr_k=THR_K,
    min_size=MIN_NUCLEOLI_SIZE,
    max_size=MAX_NUCLEOLI_SIZE,
    fill_holes=FILL_HOLES,
    closing_radius=CLOSING_RADIUS,
    opening_radius=OPENING_RADIUS,
    split_touching=SPLIT_TOUCHING,
    peak_min_dist=PEAK_MIN_DIST,
    watershed_compactness=WATERSHED_COMPACTNESS
):
    """
    Returns an image-level labeled nucleoli mask (unique IDs across nuclei).

    Strategy:
    - work per nucleus
    - segment on SMOOTHED RAW segmentation channel
    - use threshold + closing + hole fill to get nucleolar territory
    - optional watershed kept available but OFF by default
    """
    out = np.zeros_like(nuc_img, dtype=np.int32)
    next_id = 1

    smooth = gaussian(nuc_img, sigma=smooth_sigma, preserve_range=True)

    for lbl in np.unique(nucleus_labels):
        if lbl == 0:
            continue

        region = (nucleus_labels == lbl)
        vals = smooth[region]

        if vals.size < 20:
            continue

        if thr_method == "otsu":
            try:
                thr = threshold_otsu(vals)
            except ValueError:
                continue
        elif thr_method == "mad":
            med = np.median(vals)
            mad = _mad(vals)
            thr = med + thr_k * (1.4826 * mad)
        else:
            raise ValueError("THR_METHOD must be 'mad' or 'otsu'")

        binary = (smooth > thr) & region

        binary = morphology.remove_small_objects(binary, min_size=min_size)

        if closing_radius > 0:
            binary = morphology.binary_closing(binary, morphology.disk(closing_radius))

        if opening_radius > 0:
            binary = morphology.binary_opening(binary, morphology.disk(opening_radius))

        if fill_holes:
            binary = ndi.binary_fill_holes(binary)

        lab = measure.label(binary)

        if lab.max() == 0:
            continue

        keep_mask = np.zeros_like(binary, dtype=bool)
        for p in measure.regionprops(lab):
            if min_size <= p.area <= max_size:
                keep_mask[lab == p.label] = True

        if not np.any(keep_mask):
            continue

        lab = measure.label(keep_mask)

        if split_touching and lab.max() > 0:
            dist = ndi.distance_transform_edt(lab > 0)

            peaks = peak_local_max(
                dist,
                labels=(lab > 0),
                min_distance=int(peak_min_dist),
                exclude_border=False
            )

            if peaks.size > 0:
                markers = np.zeros_like(lab, dtype=np.int32)
                for i, (r, c) in enumerate(peaks, start=1):
                    markers[r, c] = i
                markers = ndi.label(markers > 0)[0]

                lab = segmentation.watershed(
                    -dist,
                    markers,
                    mask=(lab > 0),
                    compactness=watershed_compactness
                )

        for k in np.unique(lab):
            if k == 0:
                continue
            out[lab == k] = next_id
            next_id += 1

    return out


# ----------------------------
# FEATURE COLLECTION (NUCLEOLI)
# ----------------------------
def collect_nucleoli_features(image_dict):
    logger.info('collecting nucleus & nucleoli features...')
    results = []

    for name, img in image_dict.items():
        secondary_img = img["secondary"]
        primary_img = img["primary"]
        segmentation_img = img["segmentation"]
        nucleus_labels = img["mask"]

        # detect nucleoli using segmentation channel
        nucleoli_labels = detect_nucleoli_labels_for_image(
            nuc_img=segmentation_img,
            nucleus_labels=nucleus_labels
        )

        contours = measure.find_contours((nucleus_labels > 0).astype(int), 0.8)
        contour = [c for c in contours if len(c) >= 100]

        for lbl in np.unique(nucleus_labels):
            if lbl == 0:
                continue

            nuc_mask = (nucleus_labels == lbl)
            if nuc_mask.sum() == 0:
                continue

            # nucleoli within this nucleus
            nuc_nucleoli = nucleoli_labels * nuc_mask
            nuc_nucleoli = measure.label(nuc_nucleoli > 0)

            df_n = feature_extractor(nuc_nucleoli).add_prefix('nucleoli_')
            if df_n.empty:
                continue

            # object-level intensity stats from PRIMARY / SECONDARY measurement channels
            stats_rows = []
            for _, row in df_n.iterrows():
                obj_lbl = int(row['nucleoli_label'])
                o_mask = (nuc_nucleoli == obj_lbl)

                v_primary = primary_img[o_mask]
                v_secondary = secondary_img[o_mask]

                mean1 = float(np.mean(v_primary)) if v_primary.size else np.nan
                mean2 = float(np.mean(v_secondary)) if v_secondary.size else np.nan
                cv = float(np.std(v_primary) / (np.mean(v_primary) + 1e-9)) if v_primary.size else np.nan
                skew_stat = skewtest(v_primary).statistic if v_primary.size >= 8 else np.nan

                stats_rows.append({
                    "nucleoli_cv": cv,
                    "nucleoli_skew": skew_stat,
                    "nucleoli_intensity_mean": mean1,
                    "nucleoli_intensity_mean_in_coi2": mean2
                })

            df_stats = pd.DataFrame(stats_rows)
            df = pd.concat([df_n.reset_index(drop=True), df_stats], axis=1)

            # nucleus-level stats from PRIMARY / SECONDARY measurement channels
            v1_nuc = primary_img[nuc_mask]
            v2_nuc = secondary_img[nuc_mask]
            mean1_nuc = float(np.mean(v1_nuc)) if v1_nuc.size else np.nan
            std1_nuc = float(np.std(v1_nuc)) if v1_nuc.size else np.nan

            df['image_name'] = name
            df['nucleus_number'] = lbl
            df['nucleus_size'] = int(nuc_mask.sum())
            df['nucleus_std'] = std1_nuc
            df['nucleus_cv'] = float(std1_nuc / (mean1_nuc + 1e-9)) if np.isfinite(mean1_nuc) else np.nan
            df['nucleus_skew'] = skewtest(v1_nuc).statistic if v1_nuc.size >= 8 else np.nan
            df['nucleus_coi1_intensity_mean'] = mean1_nuc
            df['nucleus_coi2_intensity_mean'] = float(np.mean(v2_nuc)) if v2_nuc.size else np.nan
            df['nucleus_coords'] = [contour] * len(df)

            # save channel metadata for traceability
            df['segmentation_channel_idx'] = SEGMENTATION_CHANNEL
            df['primary_measure_channel_idx'] = PRIMARY_MEASURE_CHANNEL
            df['secondary_measure_channel_idx'] = SECONDARY_MEASURE_CHANNEL
            df['segmentation_channel_name'] = SEGMENTATION_CHANNEL_NAME
            df['primary_measure_channel_name'] = PRIMARY_MEASURE_CHANNEL_NAME
            df['secondary_measure_channel_name'] = SECONDARY_MEASURE_CHANNEL_NAME

            results.append(df)

    logger.info('feature extraction done.')
    return pd.concat(results, ignore_index=True) if results else pd.DataFrame()


def extra_nucleoli_features(df):
    """
    Add interpretable shape + heterogeneity metrics.
    """
    df = df.copy()

    df['nucleoli_aspect_ratio'] = df['nucleoli_minor_axis_length'] / (df['nucleoli_major_axis_length'] + 1e-9)
    df['nucleoli_circularity'] = (4 * np.pi * df['nucleoli_area']) / (df['nucleoli_perimeter']**2 + 1e-9)
    df['nucleoli_compactness'] = (df['nucleoli_perimeter']**2) / (4 * np.pi * df['nucleoli_area'] + 1e-9)
    df['nucleoli_equiv_diameter'] = 2.0 * np.sqrt(df['nucleoli_area'] / np.pi)

    if 'nucleoli_perimeter_crofton' in df.columns:
        df['nucleoli_perimeter_ratio'] = df['nucleoli_perimeter'] / (df['nucleoli_perimeter_crofton'] + 1e-9)
    else:
        df['nucleoli_perimeter_ratio'] = np.nan

    hu_cols = [c for c in df.columns if c.startswith("nucleoli_moments_hu-")]
    if len(hu_cols) > 0:
        for c in hu_cols:
            x = df[c].astype(float)
            df[c + "_log"] = -np.sign(x) * np.log10(np.abs(x) + 1e-30)

        log_cols = [c + "_log" for c in hu_cols]
        df["nucleoli_hu_log_norm"] = np.sqrt(np.sum(df[log_cols].to_numpy()**2, axis=1))

    df['coi2_partition_coeff'] = df['nucleoli_intensity_mean_in_coi2'] / (df['nucleus_coi2_intensity_mean'] + 1e-9)
    df['coi1_partition_coeff'] = df['nucleoli_intensity_mean'] / (df['nucleus_coi1_intensity_mean'] + 1e-9)

    df['nucleoli_area_fraction_of_nucleus'] = df['nucleoli_area'] / (df['nucleus_size'] + 1e-9)
    df['nucleoli_enrichment_coi1'] = df['nucleoli_intensity_mean'] / (df['nucleus_coi1_intensity_mean'] + 1e-9)
    df['nucleoli_mass_coi1'] = df['nucleoli_area'] * df['nucleoli_intensity_mean']

    return df


def aggregate_features_by_group(df, group_cols, agg_cols, agg_func='mean'):
    grouped_dfs = []
    for col in agg_cols:
        agg_df = df.groupby(group_cols)[col].agg(agg_func).reset_index()
        grouped_dfs.append(agg_df)

    merged_df = functools.reduce(
        lambda left, right: left.merge(right, on=group_cols),
        grouped_dfs
    )
    return merged_df.reset_index(drop=True)


def debug_nucleoli_steps(
    nuc_img,
    nucleus_labels,
    smooth_sigma=NUC_SMOOTH_SIGMA,
    thr_method=THR_METHOD,
    thr_k=THR_K,
    closing_radius=CLOSING_RADIUS,
    opening_radius=OPENING_RADIUS,
    min_size=MIN_NUCLEOLI_SIZE
):
    """
    Returns debug images:
    - smooth
    - thresholded binary (before/after cleanup)
    built across all nuclei
    """
    smooth = gaussian(nuc_img, sigma=smooth_sigma, preserve_range=True)

    binary_raw = np.zeros_like(nuc_img, dtype=bool)
    binary_clean = np.zeros_like(nuc_img, dtype=bool)

    for lbl in np.unique(nucleus_labels):
        if lbl == 0:
            continue

        region = (nucleus_labels == lbl)
        vals = smooth[region]
        if vals.size < 20:
            continue

        if thr_method == "otsu":
            try:
                thr = threshold_otsu(vals)
            except ValueError:
                continue
        elif thr_method == "mad":
            med = np.median(vals)
            mad = _mad(vals)
            thr = med + thr_k * (1.4826 * mad)
        else:
            raise ValueError("THR_METHOD must be 'mad' or 'otsu'")

        b = (smooth > thr) & region
        binary_raw |= b

        b = morphology.remove_small_objects(b, min_size=min_size)

        if closing_radius > 0:
            b = morphology.binary_closing(b, morphology.disk(closing_radius))

        if opening_radius > 0:
            b = morphology.binary_opening(b, morphology.disk(opening_radius))

        b = ndi.binary_fill_holes(b)
        binary_clean |= b

    return smooth, binary_raw, binary_clean


# --- Proof Plotting ---
def generate_proofs(df, image_dict):
    """
    5-panel proof:
      (1) Other/secondary channel raw
      (2) Primary measurement channel raw
      (3) Smoothed segmentation image
      (4) Binary threshold mask
      (5) Final survivors overlaid on primary image
    """
    logger.info('Generating proof plots...')
    for name, img in image_dict.items():
        secondary_img = img["secondary"]
        primary_img = img["primary"]
        segmentation_img = img["segmentation"]
        mask = img["mask"]

        region_img = primary_img * (mask > 0)

        contours = measure.find_contours((mask > 0).astype(int), 0.8)
        contours = [c for c in contours if len(c) >= 100]

        smooth_dbg, binary_raw_dbg, binary_clean_dbg = debug_nucleoli_steps(
            nuc_img=segmentation_img,
            nucleus_labels=mask
        )

        survivors = detect_nucleoli_labels_for_image(
            nuc_img=segmentation_img,
            nucleus_labels=mask
        )

        fig, axes = plt.subplots(1, 5, figsize=(24, 6))
        ax0, ax1, ax2, ax3, ax4 = axes

        ax0.imshow(secondary_img, cmap='gray_r')
        ax0.set_title(f'{SECONDARY_MEASURE_CHANNEL_NAME} (secondary)')
        ax0.axis('off')

        ax1.imshow(primary_img, cmap='gray_r')
        ax1.set_title(f'{PRIMARY_MEASURE_CHANNEL_NAME} raw')
        ax1.axis('off')

        ax2.imshow(smooth_dbg, cmap='gray_r')
        ax2.set_title(f'{SEGMENTATION_CHANNEL_NAME} smoothed')
        ax2.axis('off')

        ax3.imshow(binary_clean_dbg, cmap='gray')
        for line in contours:
            ax3.plot(line[:, 1], line[:, 0], c='red', lw=0.5)
        ax3.set_title('binary cleaned')
        ax3.axis('off')

        ax4.imshow(region_img, cmap='gray_r')
        for line in contours:
            ax4.plot(line[:, 1], line[:, 0], c='k', lw=0.6)

        for c in measure.find_contours((survivors > 0).astype(float), 0.5):
            ax4.plot(c[:, 1], c[:, 0], color='red', lw=1.0)

        ax4.set_title('final survivors')
        ax4.axis('off')

        scalebar = ScaleBar(
            SCALE_PX, SCALE_UNIT, location='lower right',
            pad=0.3, sep=2, box_alpha=0, color='gray',
            length_fraction=0.3
        )
        ax1.add_artist(scalebar)

        n_survive = np.unique(survivors).size - 1
        fig.suptitle(f'{name} | survivors: {n_survive}', y=0.98)
        fig.tight_layout()
        fig.savefig(f'{proofs_folder}{name}_proof.png', dpi=300, bbox_inches='tight')
        plt.close(fig)

    logger.info('proofs saved.')


if __name__ == '__main__':
    logger.info('loading images and masks...')
    images = load_images(image_folder)
    masks = load_masks(mask_folder)

    quant_masks = build_quant_masks(masks, QUANT_REGION)
    filtered = filter_saturated_images(images, quant_masks, masks)

    features = collect_nucleoli_features(filtered)
    if features.empty:
        logger.warning("No nucleoli detected after filtering; nothing to save/plot.")
        sys.exit(0)

    features = extra_nucleoli_features(features)

    generate_proofs(features, filtered)
    logger.info('proofs complete.')

    logger.info('starting data wrangling and saving...')

    cols_to_drop = [col for col in features.columns if '_coords' in col]
    features = features.drop(columns=cols_to_drop)
    features.to_csv(f'{output_folder}nucleoli_features.csv', index=False)

    logger.info('data wrangling and saving complete.')
    logger.info('pipeline complete.')