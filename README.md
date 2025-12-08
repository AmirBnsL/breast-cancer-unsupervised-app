# Breast Cancer Unsupervised Project

This project applies unsupervised machine learning techniques to a breast cancer dataset. The primary goal is to explore hidden structures within the data, group similar patients using clustering algorithms, and identify patterns related to breast cancer characteristics.

## Project Structure

The project is organized as follows:

```
breast_cancer_unsupervised_project/
│
├── data/
│   ├── raw/                # Original dataset (immutable)
│   ├── processed/          # Cleaned, scaled, and preprocessed data
│   └── external/           # Metadata and additional information
│
├── notebooks/
│   ├── 01_exploration.ipynb      # Exploratory Data Analysis (EDA)
│   ├── 02_preprocessing.ipynb    # Data cleaning, padding, and scaling
│   └── 03_modeling.ipynb         # Clustering model development and testing
│
├── src/
│   ├── __init__.py
│   ├── data_preprocessing.py     # Data cleaning and transformation scripts
│   ├── features.py               # Feature engineering and selection
│   ├── models.py                 # Clustering algorithm implementations
│   ├── evaluation.py             # Model evaluation metrics
│   └── utils.py                  # Utility helper functions
│
├── gradio_app/
│   └── app.py                    # Main Gradio application interface
│
├── models/
│   ├── trained_models/           # Serialized trained models
│   └── artifacts/                # Auxiliary artifacts (scalers, PCA models, etc.)
│
├── reports/
│   ├── figures/                  # Generated plots and visualizations
│   └── final_report.md           # Comprehensive project report
│
├── requirements.txt              # Project dependencies
├── README.md                     # Project documentation
└── .gitignore                    # Git ignore rules
```

## Getting Started

1.  **Clone the repository:**
    ```bash
    git clone <repository-url>
    cd breast_cancer_unsupervised_project
    ```

2.  **Install dependencies:**
    It is recommended to use a virtual environment.
    ```bash
    pip install -r requirements.txt
    ```

3.  **Run the Gradio App:**
    ```bash
    python gradio_app/app.py
    ```
    The application will launch in your default web browser.

## Key Features

*   **Data Preprocessing:** Handling missing values, scaling, and interpolation.
*   **Unsupervised Learning:** Application of clustering algorithms to identify patient groups.
*   **Interactive Interface:** A Gradio-based web application to explore the models and results.
