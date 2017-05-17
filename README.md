# Kaggle - Two Sigma
Repository contains model I built to solve Two Sigma competition from Kaggle.

The aim of the competition was to predict probability that interest in rental listing will reach certain level. More detailed description of the competition and the applied model can be found in notebook two-sigma-notebook.ipynb.

Python, particularly Pandas and Scikit-Learn libraries, and XGBoost were employed to solve the problem. Solution can be found in two-sigma-xgboost.py. Files predict-from-description.py and predict-from-features.py contain models used to create new features from text data. These models make use of helper function from file split-data.py.
