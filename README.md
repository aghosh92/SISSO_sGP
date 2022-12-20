# SISSO_sGP

This project introduces a novel approach for active learning over the chemical spaces based on hypothesis learning. An example is demonstrated using the QM9 dataset for predicting formation enthalpy for molecules.

We construct the hypotheses on the possible relationships between structures and functionalities of interest based on a small subset of data as shown in this notebook.
[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/aghosh92/SISSO_sGP/blob/main/physics_informed_featurization_QM9_1k_molecules.ipynb)

These are then used as analytical models (mean functions) for the Gaussian process to reconstruct functional behavior over a broader chemical space. The example workflow can be run by executing the Python script as given in this repository. The results as performed for all ~130,000 molecules from QM9 dataset can be visualized using this notebook. 



