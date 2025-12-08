# breast_cancer_unsupervised_project
This project applies unsupervised machine learning to a breast-cancer padded–interpolated dataset. The goal is to explore the hidden structure in the data, group similar patients, and help understand possible patterns related to breast cancer characteristics.

breast_cancer_unsupervised_project/
│
├── data/
│   ├── raw/                # original dataset (never modify)
│   ├── processed/          # cleaned, scaled, padded/interpolated data
│   └── external/           # any additional info (metadata, notes…)
│
├── notebooks/
│   ├── 01_exploration.ipynb      # EDA: distributions, missing values…
│   ├── 02_preprocessing.ipynb    # padding, interpolation, scaling
│   └── 03_modeling.ipynb         # testing clustering algorithms
│
├── src/
│   ├── __init__.py
│   ├── data_preprocessing.py     # functions for cleaning, scaling…
│   ├── features.py               # feature selection / PCA
│   ├── models.py                 # clustering algorithms
│   ├── evaluation.py             # metrics functions
│   └── utils.py                  # helper functions
│
├── streamlit_app/
│   ├── app.py                    # main streamlit interface
│   └── components/               # reusable UI pieces
│
├── models/
│   ├── trained_models/           # saved clustering models
│   └── artifacts/                # PCA models, scalers...
│
├── reports/
│   ├── figures/                  # plots: clusters, PCA, metrics…
│   └── final_report.md           # your final written report
│
├── requirements.txt
├── README.md
└── .gitignore
