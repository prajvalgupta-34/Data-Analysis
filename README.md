# Data Analysis & Machine Learning Projects

A collection of Jupyter notebooks covering exploratory data analysis, data visualization and classical machine learning (KNN, Decision Tree, Random Forest) in Python.

## Projects at a Glance

| # | Notebook | Topic | Type | Key Tools |
|---|----------|-------|------|-----------|
| 1 | [`Movie_Capstone-Project.ipynb`](Movie_Capstone-Project.ipynb) | Are Fandango movie ratings inflated? | EDA & Visualization | pandas, seaborn, matplotlib |
| 2 | [`Seaborn.ipynb`](Seaborn.ipynb) | Credit card applicant data plots | Visualization | pandas, seaborn, matplotlib |
| 3 | [`Student_Ai.ipynb`](Student_Ai.ipynb) | Predict students' final grade | Classification (KNN) | scikit-learn, kagglehub |
| 4 | [`food_delivery.ipynb`](food_delivery.ipynb) | Predict online food order outcome | Classification (KNN) | scikit-learn, kagglehub |
| 5 | [`stock_market.ipynb`](stock_market.ipynb) | Out-of-time share price forecasting | Regression | scikit-learn, matplotlib |

---

## 1. Movie Capstone: Fandango vs. Other Rating Sites

**Goal:** Reproduce the analysis behind FiveThirtyEight's article *"Be Suspicious Of Online Movie Ratings, Especially Fandango's"* and check whether Fandango inflates its displayed ratings.

**Data:** `fandango_scrape.csv` and `all_sites_scores.csv` from [FiveThirtyEight's data repo](https://github.com/fivethirtyeight/data) (Fandango, Rotten Tomatoes, Metacritic and IMDb scores).

**What's done:**
- Explored the relationship between votes and ratings, and extracted the release year from film titles
- Compared Fandango's displayed `STARS` against the true `RATING` using KDE plots and a difference count plot
- Compared Rotten Tomatoes critic vs. user scores, and found the movies with the biggest disagreement
- Explored Metacritic and IMDb scores and vote counts
- Merged the Fandango and all-sites tables, then normalized every score to a 0-5 scale
- Compared rating distributions across sites with KDE plots, a histogram and a clustermap
- Examined the 10 worst-rated movies (by Rotten Tomatoes) across all platforms

**Takeaway:** Fandango's displayed ratings sit noticeably higher than those of the other sites, even for films that critics rated poorly.

---

## 2. Seaborn Exercises: Credit Card Approval Data

**Goal:** Recreate a set of target plots from the credit card approval dataset to practice seaborn and pandas.

**Data:** [Credit Card Approval Prediction](https://www.kaggle.com/rikdifos/credit-card-approval-prediction) (`application_record.csv`) from Kaggle. Features include income, education, family status, housing type, days employed and occupation.

**Plots created:**
- Scatter plot of days employed vs. age (employed applicants only), colored by occupation
- Histogram of age in years, split by occupation
- Box plots of income by family status for the bottom half of earners, split by property ownership
- Correlation heatmap of the numeric columns

---

## 3. Student Performance Prediction

**Goal:** Predict a student's final grade (A-F) from their study habits and background.

**Data:** [Student Performance and Study Habits Dataset](https://www.kaggle.com/datasets/harshadapatil31/student-performance-and-study-habits-dataset) (1,000 rows, 12 columns) loaded through `kagglehub`.

**Approach:**
1. Encode categorical columns (gender, parental education, internet access, part-time job, extracurricular activities, final grade) as numbers
2. Drop rows with missing values
3. Split into train and test sets and scale features with `StandardScaler`
4. Train a `KNeighborsClassifier` (Manhattan distance, `p=1`)

**Result:** Best accuracy of **~78.7%** at `K = 39`.

---

## 4. Online Food Delivery Prediction

**Goal:** Predict whether a customer's order outcome (`Output`) is Yes or No from demographic information.

**Data:** [Online Food Ordering Dataset](https://www.kaggle.com/datasets/srisyra02/online-food-ordering-dataset) loaded through `kagglehub`.

**Approach:**
1. Encode ordinal columns (monthly income, education, customer type, feedback) with manual mappings
2. Check for missing values and duplicates, then drop duplicates
3. One-hot encode the remaining categorical features
4. Train a `KNeighborsClassifier` with `n_neighbors=5`

**Features used:** age, gender, marital status, occupation, monthly income, education, family size, customer type, latitude, longitude and pin code.

**Result:** Test accuracy of **~79.2%**.

---

## 5. Stock Market: Out-of-Time Share Price Extrapolation

**Goal:** Forecast future share prices using only past information, with a strict chronological split so there is no data leakage.

**Data:** [Stock Market Dataset](https://www.kaggle.com/datasets/jasineri/stock-market-dataset) loaded through `kagglehub`.

**Approach:**
- Auto-detect the ID, timestamp and price columns, with manual overrides available
- Build features from earlier rows only: price lags (1, 2, 3, 5), 1-step return, rolling mean (3 and 5) and rolling std (5), day of week and month
- Train on the earliest 80% of the timeline and test on the latest 20%
- Compare three regressors and plot actual vs. predicted prices for one stock

**Results:**

| Model | RMSE ($) | MAE ($) | R² |
|-------|---------:|--------:|---:|
| Decision Tree | 6181.33 | 158.86 | 0.9979 |
| **Random Forest** | **2508.10** | **76.33** | **0.9997** |
| K-Nearest Neighbors | 3994.33 | 457.08 | 0.9991 |

Random Forest gave the lowest error on all three metrics.

---

## Getting Started

### Prerequisites
- Python 3.9+
- Jupyter Notebook or JupyterLab

### Installation

```bash
git clone https://github.com/<your-username>/<your-repo>.git
cd <your-repo>
pip install pandas numpy matplotlib seaborn scikit-learn scipy kagglehub jupyter
```

### Run

```bash
jupyter notebook
```

Then open any notebook and run the cells in order.

### Data Setup

| Notebook | How to get the data |
|----------|---------------------|
| Student_Ai, food_delivery, stock_market | Downloaded automatically with `kagglehub` (may require Kaggle credentials) |
| Movie_Capstone-Project | Download `fandango_scrape.csv` and `all_sites_scores.csv` from [FiveThirtyEight](https://github.com/fivethirtyeight/data) and place them next to the notebook |
| Seaborn | Download `application_record.csv` from [Kaggle](https://www.kaggle.com/rikdifos/credit-card-approval-prediction) and place it next to the notebook |

## Tech Stack

Python · pandas · NumPy · Matplotlib · seaborn · scikit-learn · SciPy · kagglehub · Jupyter

## Repository Structure

```
.
├── Movie_Capstone-Project.ipynb
├── Seaborn.ipynb
├── Student_Ai.ipynb
├── food_delivery.ipynb
├── stock_market.ipynb
└── README.md
```

## Acknowledgements

- [FiveThirtyEight](https://github.com/fivethirtyeight/data) for the Fandango data
- Pierian Data for the Seaborn exercise template and Movie Capstone project outline
- Kaggle dataset authors for the credit card, student, food ordering and stock market datasets

## License

Add a license of your choice (for example MIT) before publishing.
