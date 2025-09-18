# Final Project – Machine Learning 
## [PROJECT LINK](https://colab.research.google.com/drive/1UvzMjHAPBsmNmUFH60CP_ZuiBP696bLi?usp=sharing) 

## 📌 Overview
This project applies **Machine Learning techniques** to analyze data, uncover insights, and build predictive models.  
The goal is to demonstrate end-to-end skills in **data preprocessing, exploratory data analysis (EDA), model training, and evaluation**, making it suitable for showcasing in a professional portfolio.

## 📊 Dataset
- **Source**: Kaggle  
- **Description**: The dataset contains information on

## 📊 Dataset Schema

| Column             | Non-Null Count | Data Type |
|--------------------|----------------|-----------|
| id                 | 560            | object    |
| city               | 560            | object    |
| country            | 560            | object    |
| region             | 560            | object    |
| short_description  | 560            | object    |
| latitude           | 560            | float64   |
| longitude          | 560            | float64   |
| avg_temp_monthly   | 560            | object    |
| ideal_durations    | 560            | object    |
| budget_level       | 560            | object    |
| culture            | 560            | int64     |
| adventure          | 560            | int64     |
| nature             | 560            | int64     |
| beaches            | 560            | int64     |
| nightlife          | 560            | int64     |
| cuisine            | 560            | int64     |
| wellness           | 560            | int64     |
| urban              | 560            | int64     |
| seclusion          | 560            | int64     |

- **Size**: (560, 19)

## 🔎 Methodology
1. **Data Preprocessing**  
   - Handling missing values  
   - Feature engineering & encoding  
   - Scaling / normalization  

2. **Exploratory Data Analysis (EDA)**  
   - Statistical summary of data  
   - Visualization of distributions and correlations  

3. **Modeling**  
   - Applied multiple supervised ML algorithms  
   - Evaluated performance using metrics like Accuracy, RMSE, R²  

4. **Evaluation**  
   - Compared models and selected the best-performing one  
 
## 📈 Results

### Random Forest Classifier
- **Accuracy:** 0.7411  

**Classification Report:**

| Class      | Precision | Recall | F1-Score | Support |
|------------|-----------|--------|----------|---------|
| Budget     | 0.66      | 0.66   | 0.66     | 29      |
| Luxury     | 0.75      | 0.40   | 0.52     | 15      |
| Mid-range  | 0.77      | 0.85   | 0.81     | 68      |

**Overall Metrics:**
- **Accuracy:** 74%  
- **Macro Avg (Precision/Recall/F1):** 0.73 / 0.64 / 0.66  
- **Weighted Avg (Precision/Recall/F1):** 0.74 / 0.74 / 0.73  



# Documentation


## 1. Project Title and Domain

- **PROJECT TITLE:** Worldwide Travel Cities (Ratings and Climate)
- **Domain:** Travel & Tourism Analytics

## 2. Objective

The primary objective is to create a data-driven travel recommendation and analysis system. This system will assist users in discovering and comparing global cities by considering various factors:

- Preferences in themes (culture, adventure, cuisine, etc.)
- Travel budget
- Ideal trip duration
- Seasonal climate conditions

The goal is to facilitate personalized, efficient, and enjoyable trip planning.

## 3. Dataset Information

- **Source:** Kaggle
- **Time Range:** 2010 – 2025
- **Total Cities:** 560+

## 4. Type of Problem

- **Classification Problem:** The project focuses on predicting categorical outcomes based on city features.
- **Target Variable:** `budget_level` (which can be 'Budget', 'Mid-range', or 'Luxury').
- **Algorithms Used:**
    - Random Forest Classifier
    - Logistic Regression
    - Support Vector Machine (SVM)
    - K-Nearest Neighbors (KNN)

## 5. Workflow Stages

### STAGE 1: Initial EDA (Exploratory Data Analysis)

- Load the dataset from the provided URL using pandas.
- Display the first few rows of the DataFrame (`df.head()`).
- Get a concise summary of the DataFrame, including data types and non-null values (`df.info()`).
- Generate descriptive statistics for numerical columns (`df.describe()`).
- List the column names (`df.columns`).

### STAGE 2: EDA (Visualization) and Pre-processing

#### Handling Missing Values

- Calculate the sum of missing values for each column (`df.isnull().sum()`).
- Identify and print the top 10 countries based on city count.

#### Handling Duplicates

- Check for duplicate rows in the dataset (`df.duplicated().sum()`).
- Remove duplicate rows (`df = df.drop_duplicates()`).

#### Outlier Detection and Treatment

- Visualize outliers using boxplots for selected columns (e.g., 'cuisine').
- Identify numerical columns.
- Apply the Interquartile Range (IQR) method to detect and treat outliers by capping values at the lower and upper bounds (`Q1 - 1.5*IQR`, `Q3 + 1.5*IQR`).
- Print a confirmation message after removing duplicates and treating outliers.
- Re-visualize boxplots for features and the target variable ('cuisine' in the example) after outlier treatment to observe the changes.

#### Skewness

- Calculate the skewness of the target variable (e.g., 'cuisine').
- Apply transformations (log1p or square root) to numerical columns with significant skewness (> 0.5 or < -0.5) to make distributions more symmetrical. Use log1p for positive values and square root (with shifting for non-positive) otherwise.

#### EDA after Preprocessing

- Re-examine `df.info()` and `df.describe()` to see the changes after preprocessing steps.
- Check the DataFrame shape (`df.shape`).

#### Visualizations

- **Univariate Analysis:** Create histograms for numerical feature variables to understand their distributions.
- **Bivariate Analysis:** Create scatter plots to visualize the relationship between each feature and the target variable (e.g., 'cuisine').
- **Multivariate Analysis:** Use pairplots for a subset of features and the target to visualize relationships and distributions across multiple variables.
- Analyze skewness before and after applying log transformation to observe the effect on distributions.
- Generate summary statistics for log-transformed features and the target.
- Create interaction terms between relevant features (e.g., 'culture' and 'nightlife').
- Visualize the relationship between the interaction term and the target variable.

### STAGE 3: Feature Engineering

- List the columns in the DataFrame.
- Create new features based on domain knowledge or combinations of existing features:
    - `culture_nightlife`: Interaction term (product of culture and nightlife scores).
    - `nature_to_urban`: Ratio of nature to urban scores.
    - `total_theme_score`: Sum of all theme scores.
    - `trip_intensity`: Categorical feature based on `total_theme_score` (Low, Medium, High, Extreme) created using `pd.cut`.
- Encode the `trip_intensity` categorical feature into numerical format using `LabelEncoder`.
- Preview the newly created features.

### STAGE 4: Feature Selection

- Select a target variable (e.g., 'cuisine' for demonstration, but the main target is `budget_level`).
- Prepare only numeric features by dropping non-numeric and constant columns.
- **Correlation Matrix:** Calculate the correlation matrix between numerical features and the target. Select features with absolute correlation greater than a specified threshold (e.g., 0.2).
- **Chi² Feature Selection:**
    - Scale numeric features using `MinMaxScaler`.
    - Convert the target variable into categories suitable for Chi² (e.g., by binning).
    - Use `SelectKBest` with the `chi2` score function to select the top k features.
- **ANOVA F-Test:**
    - Use `SelectKBest` with the `f_classif` score function to select the top k features.
- Print the lists of features selected by each method.

### STAGE 5: Model Building (Classification)

- Define the classification target variable (`budget_level`).
- Prepare features (`X_classification`) by selecting numerical columns from the DataFrame, ensuring engineered features are included and the target is excluded. Handle potential missing values by filling with the mean.
- Prepare the target variable (`y_classification`).
- Encode the categorical target variable into numerical format using `LabelEncoder`.
- Scale the features using `MinMaxScaler`.
- Split the data into training and testing sets (`X_train_clf`, `X_test_clf`, `y_train_clf`, `y_test_clf`) using `train_test_split`, ensuring stratification based on the target variable.

#### Train and Evaluate Models

- **Random Forest Classifier:**
    - Initialize and train a `RandomForestClassifier` model.
    - Make predictions on the test set.
    - Evaluate the model using `accuracy_score` and `classification_report`. Print the results.
- **Logistic Regression:**
    - Initialize and train a `LogisticRegression` model (adjust `max_iter` if needed for convergence).
    - Make predictions on the test set.
    - Evaluate the model using `accuracy_score` and `classification_report`. Print the results.
- **Support Vector Machine (SVM):**
    - Initialize and train an `SVC` model (set `probability=True` if needed).
    - Make predictions on the test set.
    - Evaluate the model using `accuracy_score` and `classification_report`. Print the results.
- **K-Nearest Neighbors (KNN):**
    - Initialize and train a `KNeighborsClassifier` model (choose `n_neighbors`).
    - Make predictions on the test set.
    - Evaluate the model using `accuracy_score` and `classification_report`. Print the results.

#### Model Comparison

- Print a comprehensive comparison of the accuracy scores for all trained classification models (Random Forest, Logistic Regression, SVM, KNN).
- Identify and print the model with the highest accuracy based on the test set evaluation.
