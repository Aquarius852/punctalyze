## Fluorescence image analyses
This repository contains image analysis scripts used in the Boeynaems lab at the Baylor College of Medicine in Houston, TX, USA. These scripts were adapted by Sovanny Taylor and Olivia M. S. Carmo for analysis of intracellular puncta to compare puncta features between conditions, and calculate other useful information like partitioning coefficient. The structure and strategy was adapted from previous work led by Olivia M. S. Carmo and Dezerae Cox<sup id="a1">[1](#f1)</sup> [![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.6747921.svg)](https://github.com/ocarmo/EMP1-trafficking_PTP7-analysis/tree/v1.1)

This specific version of punctalyze was forked and modified for use in the Fazal Lab by Kyle Nielson.

### Analysis software
This repo relies extensively on the following python packages: [CellPose](https://www.cellpose.org/)<sup id="a2">[2](#f2)</sup>, [napari](https://napari.org/)<sup id="a3">[3](#f3)</sup>, and [scikit-image](https://scikit-image.org/)<sup id="a4">[4](#f4)</sup>

### Reproducing workflow
See the (associated template)[https://github.com/ocarmo/punctalyze-template] README.md for setup instructions. Run the scripts in the order indicated by the script prefix. To create a suitable conda environment, run each line of the ```environment_setup.md``` in succession in your command prompt. 

## References
<b id="f1">1.</b> This repository format adapted from https://github.com/ocarmo/EMP1-trafficking_PTP7-analysis [↩](#a1)

<b id="f2">2.</b> Pachitariu M, Rariden M, Stringer C. Cellpose-SAM: superhuman generalization for cellular segmentation. biorxiv. 2025; doi:10.1101/2025.04.28.651001. [↩](#a2)

<b id="f3">3.</b> Sofroniew N, Lambert T, Evans K, Nunez-Iglesias J, Winston P, Bokota G, et al. napari/napari: 0.4.9rc2. Zenodo; 2021. doi:10.5281/zenodo.4915656. [↩](#a3)

<b id="f4">4.</b> Walt S van der, Schönberger JL, Nunez-Iglesias J, Boulogne F, Warner JD, Yager N, et al. scikit-image: image processing in Python. PeerJ. 2014;2: e453. doi:10.7717/peerj.453. [↩](#a4)
