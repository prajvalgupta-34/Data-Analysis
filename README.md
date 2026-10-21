# Data Analysis Projects

These are the notebooks I've built while learning data analysis and machine learning in Python. Some are visualization practice, some are small ML models on Kaggle datasets.

## What's inside

- **Movie_Capstone-Project.ipynb**: checking whether Fandango's movie ratings are higher than other sites
- **Seaborn.ipynb**: seaborn plot exercises on a credit card applicant dataset
- **Student_Ai.ipynb**: predicting a student's final grade with KNN
- **food_delivery.ipynb**: predicting online food order outcomes with KNN
- **stock_market.ipynb**: forecasting share prices with Decision Tree, Random Forest and KNN

## Movie Capstone

This one follows FiveThirtyEight's article on Fandango's ratings. I compared the stars Fandango shows against the actual ratings, then lined it up with Rotten Tomatoes, Metacritic and IMDb after scaling every score to 0-5.

Fandango came out higher than the other sites, even for movies critics didn't like. The data comes from [FiveThirtyEight's GitHub](https://github.com/fivethirtyeight/data).

## Seaborn Exercises

I used the [Credit Card Approval](https://www.kaggle.com/rikdifos/credit-card-approval-prediction) dataset to recreate a few given plots: a scatter plot of age against days employed, an age histogram by occupation, box plots of income by family status, and a correlation heatmap.

## Student Grade Prediction

Uses the [Student Performance and Study Habits](https://www.kaggle.com/datasets/harshadapatil31/student-performance-and-study-habits-dataset) dataset (1,000 students). I converted the text columns to numbers, scaled the features and trained a KNN classifier. With K=39 it reached about 78.7% accuracy.

## Food Delivery Prediction

Uses the [Online Food Ordering](https://www.kaggle.com/datasets/srisyra02/online-food-ordering-dataset) dataset. After encoding the columns and removing duplicates, a KNN classifier with 5 neighbors got about 79.2% accuracy on the test set.

## Stock Market Forecasting

Uses the [Stock Market Dataset](https://www.kaggle.com/datasets/jasineri/stock-market-dataset). I built features from past prices only (lags, rolling averages, day and month), trained on the first 80% of the timeline and tested on the last 20% so the model never sees the future.

| Model | RMSE | MAE | R² |
|---|---|---|---|
| Decision Tree | 6181.33 | 158.86 | 0.9979 |
| Random Forest | 2508.10 | 76.33 | 0.9997 |
| KNN | 3994.33 | 457.08 | 0.9991 |

Random Forest did best on all three. The R² values are very high partly because yesterday's price is already a strong hint for today's.

## Running the notebooks

You'll need Python 3 and Jupyter.

```bash
pip install pandas numpy matplotlib seaborn scikit-learn scipy kagglehub jupyter
jupyter notebook
```

The Student, Food Delivery and Stock Market notebooks download their data through `kagglehub`, which may ask for your Kaggle login. For the Movie and Seaborn notebooks, download the CSV files from the links above and put them in the same folder as the notebook.
