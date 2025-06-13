# EDA and Logistic Regression on TITANIC DATASET

![image](https://github.com/user-attachments/assets/b42b357d-0bd5-4220-80c0-d1dcef27bed4)


## Titanic Survival Prediction: Exploratory Data Analysis & Logistic Regression
This repository contains a Jupyter Notebook " TITANIC_EDA_EMM.ipynb " that performs a comprehensive Exploratory Data Analysis (EDA) on the famous Titanic dataset, followed by building and evaluating a Logistic Regression model to predict passenger survival.

## 🚢 Project Introduction
The Titanic dataset is a classic for introducing machine learning concepts. It contains information about passengers aboard the ill-fated RMS Titanic, including their age, sex, passenger class, fare, and whether they survived. The goal of this project is to understand the factors that influenced survival and to build a predictive model.

## 📊 Dataset
The dataset used is titanic.csv, commonly available on Kaggle. It contains the following columns:

* PassengerId: Unique ID for each passenger.

* Survived: Survival (0 = No, 1 = Yes) - Target Variable.

* Pclass: Passenger Class (1 = 1st, 2 = 2nd, 3 = 3rd).

* Name: Passenger's name.

* Sex: Passenger's sex (male/female).

* Age: Age in years.

* SibSp: Number of siblings/spouses aboard.

* Parch: Number of parents/children aboard.

* Ticket: Ticket number.

* Fare: Passenger fare.

* Cabin: Cabin number.

* Embarked: Port of embarkation (C = Cherbourg, Q = Queenstown, S = Southampton).

## 🔍 Exploratory Data Analysis (EDA)
The EDA phase focused on understanding the data's structure, identifying missing values, exploring distributions, and discovering relationships between features and the target variable (Survived).

****Key EDA Steps & Insights:****

**1.Initial Inspection:**

* Identified 891 rows and 12 columns.

* Noted missing values in Age, Cabin, and Embarked.

* Observed that approximately 38% of passengers survived.

**2.Missing Value Handling:**

* Cabin: Dropped due to a high percentage of missing values (over 77%).

* Age: Imputed with the median age.

* Embarked: Imputed with the mode (most frequent port).

**3.Univariate Analysis:**

* Sex: More males than females were on board.

* Pclass: Most passengers were in 3rd class.

* Fare: Highly skewed, with most fares being low but some very high.

* Age: Roughly normally distributed, with a peak around 20-30 years.

* SibSp & Parch: A large majority traveled alone or with very few family members.

**4. Bivariate & Multivariate Analysis:**

* "Women and Children First": Confirmed by a significantly higher survival rate for females and young children.

* Socio-economic Status Matters: Higher Pclass (1st class) and higher Fare strongly correlated with higher survival rates.

* Family Size Impact: Passengers with small to medium FamilySize (2-4 members) had better survival rates than those traveling alone or in very large families.

* Embarked Port: Passengers from Cherbourg ('C') showed a higher survival rate, possibly due to a higher proportion of first-class passengers.

**5. Feature Engineering:**

* Created FamilySize = SibSp + Parch + 1.

* Created IsAlone (1 if FamilySize is 1, else 0). This revealed that being alone generally lowered survival chances.

## 🤖 Machine Learning Model: Logistic Regression
A Logistic Regression model was chosen for its interpretability and effectiveness in binary classification tasks.

**Model Pipeline:**
To ensure robust and reproducible preprocessing, a scikit-learn pipeline was constructed:

**1. ColumnTransformer:**

* Numerical Features (Age, Fare, SibSp, Parch, FamilySize): Scaled using StandardScaler to bring them to a similar range.

* Categorical Features (Pclass, Sex, Embarked, IsAlone): Converted into numerical format using OneHotEncoder.

**2.LogisticRegression Classifier:** The preprocessed features are then fed into the logistic regression model.

**Training and Evaluation:**

*The data was split into an 80% training set and a 20% testing set, with stratification to maintain class balance.

* The model was trained on the training data.

  **Evaluation Metrics:**

* Accuracy: Achieved approximately 81% on the test set.

* Confusion Matrix: Provided insights into True Positives, True Negatives, False Positives, and False Negatives.

* Classification Report: Detailed precision, recall, and F1-score for both 'Survived' and 'Not Survived' classes.

* ROC Curve & AUC Score: An AUC score of approximately 0.87 indicates good discriminative power of the model.

**Feature Importance (Model Coefficients):**

Interpreting the logistic regression coefficients (after scaling and encoding) reveals the most impactful features on survival:

* Sex_male (Negative Coefficient): Being male was the strongest negative predictor of survival.

* Sex_female (Positive Coefficient): Being female was a strong positive predictor of survival.

* Pclass_1 (Positive Coefficient): Being in 1st class significantly increased the odds of survival.

* Age (Negative Coefficient): Generally, older age decreased the odds of survival.

* FamilySize (Negative Coefficient): Larger family sizes tended to decrease the odds of survival, particularly very large ones.

* Fare (Positive Coefficient): Higher fare increased the odds of survival, correlating with Pclass.

## 🎉 Results and Conclusion
The Logistic Regression model effectively captured the key drivers of survival on the Titanic, aligning strongly with historical accounts and intuitive understanding (e.g., "women and children first," importance of social class). The model achieved a respectable accuracy and AUC score, demonstrating its predictive capability.

## 🚀 Future Work / Improvements
**More Advanced Feature Engineering:**

* Extracting titles from Name (e.g., Mr., Mrs., Master, Miss) could be a powerful categorical feature.

* Analyzing Ticket prefixes for patterns.

* Deriving deck information from Cabin if missing values are handled differently.

  **Hyperparameter Tuning:** Optimize LogisticRegression parameters (e.g., C for regularization) using GridSearchCV or RandomizedSearchCV.

* Other Models: Experiment with more complex models like Random Forests, Gradient Boosting Machines (XGBoost, LightGBM), or Support Vector Machines for potential performance gains.

* Ensemble Methods: Combine predictions from multiple models.

* Cross-validation: Use k-fold cross-validation for more robust evaluation.

## 💻 How to Run the Notebook

* Install dependencies:

" pip install pandas numpy scikit-learn matplotlib seaborn jupyter "

* Download the Dataset: Ensure titanic.csv is in the same directory as the notebook. You can usually find it on Kaggle's Titanic competition page.

* Launch Jupyter Notebook:

jupyter notebook

* Open and run TITANIC_EDA_EMM.ipynb in your browser.


## 🛠️ Technologies Used

* Python 3.x

* Jupyter Notebook

* Pandas (for data manipulation)

* NumPy (for numerical operations)

* Matplotlib (for plotting)

* Seaborn (for statistical visualizations)

* Scikit-learn (for machine learning models and preprocessing)
