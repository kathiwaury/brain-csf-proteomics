# Deciphering protein secretion from the brain to cerebrospinal fluid for biomarker discovery
### Katharina Waury, Renske de Wit, Inge M.W. Verberk, Charlotte E. Teunissen and Sanne Abeln

We have collected discovery proteomics studies based on untargeted mass spectrometry of healthy cerebrospinal fluid (CSF). The curated CSF proteome was used to annotate the 
[Human Protein Atlas](https://www.proteinatlas.org/humanproteome/brain/human+brain) brain elevated proteins regarding CSF presence. Based on CSF and non-CSF brain proteins two logistic classifier 
models were trained: the full CSF model and the high confidence CSF model. These models were applied to the Human Protein Atlas brain elevated proteome and a set of novel CSF proteins identified by 
proximity extension assays instead of mass spectrometry. The predicted proability for protein secretion to CSF for the human proteome is provided to support biomarker research.

![](https://github.com/kathiwaury/brain-csf-proteomics/blob/main/Workflow_overview.png)

This repository contains all data and code to train the CSF-specific protein secretion predictor and to reproduce the figures of the manuscript. The manuscript is currently under review.

The trained models can be found [here](https://github.com/kathiwaury/brain-csf-proteomics/tree/main/Models).
The prediction scores for the human proteome can be downloaded [here](https://github.com/kathiwaury/brain-csf-proteomics/blob/main/Datasets/Biomarker_discovery/Probability_scores_human_proteome.xlsx).

# Citation
If you use the the analysis or prediction scores, please cite the following publication:

Deciphering Protein Secretion from the Brain to Cerebrospinal Fluid for Biomarker Discovery (2023) Katharina Waury, Renske de Wit, Inge M. W. Verberk, Charlotte E. Teunissen, and Sanne Abeln
Journal of Proteome Research. 22(9), 3068-3080. DOI: [10.1021/acs.jproteome.3c00366](https://doi.org/10.1021/acs.jproteome.3c00366)
