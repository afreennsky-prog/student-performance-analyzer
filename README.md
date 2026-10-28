# Student Performance Analyzer

Can we predict a student's final math grade, and how much of that prediction comes only from earlier grades? A small, reproducible analysis of the UCI *Student Performance* dataset using pandas and scikit-learn.

<!-- TODO: replace the line below with your own screenshot, e.g. !\[Notebook screenshot](reports/figures/notebook\_screenshot.png) -->

**Screenshot:** *add a screenshot of the notebook or a key figure here before publishing.*

## Problem

Given a student's background, habits and earlier grades, predict the final grade **G3** (0-20) of a math course, and whether the student passes (**G3 >= 10**). The main question is a practical one: *how useful is the prediction before the first and second period grades (G1, G2) exist?* Those are the moments when a teacher could still help.

## Dataset

* **Source:** UCI Machine Learning Repository, *Student Performance*, math course file `student-mat.csv` (395 students, 33 columns, no missing values, semicolon-separated).
* **Citation:** P. Cortez and A. Silva. *Using Data Mining to Predict Secondary School Student Performance.* In A. Brito and J. Teixeira (Eds.), Proceedings of 5th FUture BUsiness TEChnology Conference (FUBUTEC 2008), pp. 5-12, Porto, Portugal, April 2008, EUROSIS, ISBN 978-9077381-39-7.
* Students come from two Portuguese secondary schools. G1, G2, G3 are the grades of the 1st, 2nd and final period.

## Method

1. **Data check and EDA:** shape, missing values, types; grade distribution, absences, study time, correlation heatmap.
2. **Preprocessing:** one-hot encoding of the 17 text columns inside a scikit-learn `Pipeline`; 80/20 train/test split, stratified on pass/fail, seed 42.
3. **Regression (G3):** mean baseline, Linear Regression, Random Forest. 5-fold cross-validation on the training set (MAE, RMSE, R2), then one final score on the held-out test set.
4. **G1/G2 check:** every model run with and without G1 and G2.
5. **Classification (pass/fail):** Random Forest, compared with a "predict pass for everyone" baseline.
6. **Interpretation:** permutation importance on the test set for the model without G1/G2.

No hyperparameter tuning was done, on purpose: with 395 rows it mostly fits noise.

## Results

**Regression: predicting G3** (5-fold CV on the 316 training students, mean ± sd across folds; last column is the 79-student test set)

|Features|Model|CV MAE|CV RMSE|CV R2|Test R2|
|-|-|-|-|-|-|
|With G1+G2|Mean baseline|3.45 ± 0.42|4.58 ± 0.45|0.00 ± 0.00|0.00|
|With G1+G2|Linear Regression|1.40 ± 0.11|2.05 ± 0.19|0.79 ± 0.08|0.77|
|With G1+G2|Random Forest|1.01 ± 0.14|1.58 ± 0.30|0.87 ± 0.06|0.88|
|Without G1+G2|Mean baseline|3.45 ± 0.42|4.58 ± 0.45|0.00 ± 0.00|0.00|
|Without G1+G2|Linear Regression|3.28 ± 0.20|4.32 ± 0.32|0.07 ± 0.25|-0.09|
|Without G1+G2|Random Forest|2.96 ± 0.38|3.86 ± 0.41|0.29 ± 0.06|0.27|

**Pass/fail classification** (79-student test set, "pass" is the positive class)

|Model|Accuracy|Precision|Recall|F1|
|-|-|-|-|-|
|Majority-class baseline (always "pass")|0.671|0.671|1.000|0.803|
|Random Forest, without G1+G2|0.646|0.687|0.868|0.767|
|Random Forest, with G1+G2|0.886|0.958|0.868|0.911|

Without G1+G2 the classifier catches 5 of the 26 failing test students; with G1+G2 it catches 24 of 26.

!\[Confusion matrices](reports/figures/05\_confusion\_matrices.png)
!\[Permutation importance](reports/figures/06\_permutation\_importance.png)

## Key findings

1. **Earlier grades do almost all the work.** Random Forest reaches R2 0.87 with G1/G2 and only 0.29 without them (Linear Regression: 0.79 vs 0.07).
2. **Without G1/G2 the prediction is weak.** Typical error is about 3 grade points on a 0-20 scale, against 3.45 for always guessing the mean.
3. **The early-warning classifier is not reliable.** Without G1/G2 its test accuracy (0.65) is below the 0.67 of predicting "pass" for everyone. The cross-validated accuracy is 0.72 ± 0.05, so a fair reading is "around baseline"; with 79 test students one split is noisy.
4. **`failures` is the clearest real signal.** Mean G3 is 11.3 for students with no past failures and 5.7 for students with three.
5. **`absences` needs care.** Its overall correlation with G3 is about zero (0.03) and it tops the importance chart, yet this is largely a data quirk: all 38 students with G3 = 0 have absences = 0. Without them the correlation is -0.21 and `absences` drops out of the top 5 in the importance check.

## How to run

```bash
git clone https://github.com/afreennsky-prog/student-performance-analyzer.git
cd student-performance-analyzer
python -m venv .venv
source .venv/bin/activate        # Windows: .venv\\Scripts\\activate
pip install -r requirements.txt
jupyter lab notebooks/analysis.ipynb   # then Run > Run All Cells
```

The notebook loads `data/student-mat.csv` and downloads it automatically if missing (`python data/download\_data.py` does the same). The downloader tries the official UCI locations first and falls back to a public GitHub copy of the same file. Figures are written to `reports/figures/`, and the regression table to `reports/results\_table.csv`.

All randomness uses seed 42. Numbers in this README come from Python 3.12, pandas 3.0, NumPy 2.4, scikit-learn 1.8, matplotlib 3.10 and seaborn 0.13; a second environment (Python 3.12, pandas 2.2.3, NumPy 1.26.4, scikit-learn 1.5.2) gave identical numbers. Older or very different versions are untested.

## Project structure

```
student-performance-analyzer/
├── data/                 student-mat.csv and download\_data.py
├── notebooks/analysis.ipynb
├── reports/
│   ├── figures/          6 saved figures
│   └── results\_table.csv
├── requirements.txt
├── LICENSE
└── README.md
```

## Limitations

* **Small data:** 395 students (316 train, 79 test). Small differences in scores are within noise.
* **One setting:** two Portuguese schools, math only, data from around 2005-2006. Results do not automatically transfer to other countries, subjects or years.
* **Correlation is not causation:** a column that predicts grades is a signal, not proof of a cause.
* **Zero grades:** 38 students have G3 = 0 and all of them have absences = 0, which looks like a recording quirk. They are kept in the data and are hard to predict.
* **Sensitive attributes:** `sex`, `age`, parents' education and jobs, `address`, `famsize`, `Pstatus`, `romantic`. These describe background, not effort. They should not be used to make decisions about real students (who gets support, attention or opportunities), because a model can reproduce unfair patterns from past data.
* **Not a deployable tool:** no fairness audit, no calibration, no monitoring.

## Future work

* Streamlit app to explore predictions for a hypothetical student.
* SHAP values for per-student explanations.
* An at-risk student report built on the no-G1/G2 model (only after a fairness review).
* Repeat the analysis without the zero-grade students, and on the Portuguese-language file.

## License

Code: MIT (see `LICENSE`). The dataset belongs to its authors; see the UCI page for its terms.

