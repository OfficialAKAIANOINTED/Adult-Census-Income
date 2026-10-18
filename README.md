# Income Prediction and Customer Segmentation (Adult Census Income)

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/OfficialAKAIANOINTED/Adult-Census-Income/blob/main/Adult_Census_Income.ipynb)

**Question:** Can we predict whether a person earns more than $50K a year from census attributes, and are there distinct population segments that a business could target?

**Data:** [UCI Adult Census Income](https://www.kaggle.com/datasets/uciml/adult-census-income) (Kaggle): 32,561 records, 15 columns. After removing 24 duplicates, 32,537 records were used. The target is imbalanced: about 24% earn over $50K.

## Approach

1. **SQL exploration:** loaded the data into SQLite and queried income by age, working hours, education and sex.
2. **Cleaning:** converted hidden `?` values to missing, filled the 3 affected categorical columns (workclass, occupation, native country) with the mode, and removed duplicates.
3. **Feature engineering:** `is_married`, `net_capital` (log-transformed), `is_us_native` and age groups. One-hot encoding, stratified 80/20 train-test split and standard scaling.
4. **Clustering:** PCA (85% of variance kept), then K-Means. K = 4 was chosen from the elbow curve and silhouette scores.
5. **Classification:** Logistic Regression, Random Forest, Gradient Boosting and SVM, evaluated on accuracy, precision, recall, F1 and ROC AUC.
6. **Imbalance handling:** class weights and SMOTE compared, then a cost-based decision threshold, plus 5-fold stratified cross-validation.

## Results

**Recommended model: Random Forest with class weights**

| Metric | Baseline RF | RF + class weights | RF + SMOTE |
|---|---|---|---|
| Accuracy | 85.25% | **85.49%** | 84.91% |
| Precision | 73.03% | **74.30%** | 71.64% |
| Recall | 61.48% | 60.84% | **61.86%** |
| F1-score | 66.76% | **66.90%** | 66.39% |
| ROC AUC | 89.89% | 89.76% | 89.49% |

**4 personas from K-Means clustering**

| Persona | Share of population | Earn over $50K |
|---|---|---|
| High-income executives and managers | 2.1% | **77.6%** |
| Mid-career professionals | 27.6% | 40.3% |
| Stable blue-collar workers | 28.3% | 33.6% |
| Entry-level / student population | 41.9% | 4.3% |

## Key findings

- One small segment (2.1% of people) has a 77.6% high-income rate, 3.2 times the overall average of 24.1%.
- Income follows career stage. Age, education, marital status and working hours together separate the segments better than any single feature.
- Accuracy alone is misleading on imbalanced data, so the model was chosen on F1 and on business cost (a missed high earner was costed at 10 times a wasted contact).

## Limitations and next steps

- Census data is from 1994 and is US-only, so the results describe that population.
- Next steps: hyperparameter tuning, XGBoost or LightGBM, and feature selection to simplify the model.

## How to run

Open the notebook in Google Colab and run all cells. The dataset downloads with `kagglehub`.

**Tools:** Python, Pandas, SQLite, Scikit-learn, imbalanced-learn, Matplotlib, Seaborn

**Author:** Anointed Akai · [LinkedIn](https://www.linkedin.com/in/akaianointed)
