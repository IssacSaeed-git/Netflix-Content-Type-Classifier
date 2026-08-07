# 🎬 Netflix Titles Type Prediction using Machine Learning

A complete Machine Learning project for predicting whether a Netflix title is a **Movie** or a **TV Show** based on extracted and engineered features from the Netflix Titles dataset.

The project covers the complete Machine Learning workflow, starting from **data exploration and preprocessing**, through **feature engineering and selection**, model training and evaluation, and finally **model deployment using Streamlit**.

---

## 📌 Project Overview

The main objective of this project is to build a classification system that predicts the type of a Netflix title:

* 🎬 **Movie**
* 📺 **TV Show**

The project uses the Netflix Titles dataset containing **8,807 records and 12 original columns**.

The original dataset contains information such as:

* `show_id`
* `type`
* `title`
* `directors`
* `cast`
* `countries`
* `date_added`
* `release_year`
* `rating`
* `duration`
* `listed_in`
* `description`

The target variable is:

```text
type
```

which represents whether the content is a **Movie** or **TV Show**.

---

# 🧠 Machine Learning Workflow

The project follows a complete Machine Learning pipeline:

```text
Netflix Dataset
      ↓
Data Loading
      ↓
EDA & Data Understanding
      ↓
Data Cleaning
      ↓
Missing Value Handling
      ↓
Duplicate Removal
      ↓
Feature Engineering
      ↓
Encoding
      ↓
Correlation Analysis
      ↓
Feature Selection
      ↓
RFE
      ↓
Train / Test Split
      ↓
Feature Scaling
      ↓
PCA
      ↓
Model Training
      ↓
Model Evaluation
      ↓
Cross Validation
      ↓
Model Comparison
      ↓
Best Model Selection
      ↓
Model Saving
      ↓
Streamlit Deployment
```

---

# 📊 Exploratory Data Analysis (EDA)

The project starts by exploring and understanding the dataset.

Some of the main operations include:

```python
df.head()
```

to inspect the first records.

```python
df.shape
```

to determine the number of rows and columns.

The dataset contains:

```text
8,807 rows
12 columns
```

The project also checks:

* Column names
* Data types
* Missing values
* Duplicate records
* Statistical information
* Target distribution

For example:

```python
df.info()
```

is used to inspect data types and missing values.

The project also uses:

```python
df.describe(include='all')
```

to generate descriptive statistics.

---

# 🧹 Data Cleaning & Preprocessing

Data preprocessing is an important part of the project because the original Netflix dataset contains missing values and some inconsistent records.

## Missing Values

The project checks missing values using:

```python
df.isna().sum()
```

Missing values are handled using different strategies depending on the column.

For example:

```python
df['directors'] = df['directors'].fillna('Unknown')
df['cast'] = df['cast'].fillna('Unknown')
df['countries'] = df['countries'].fillna('Unknown')
```

Categorical columns such as `rating` and `duration` are filled using their mode.

The `date_added` column is handled using forward filling.

---

## 🔧 Data Quality Correction

The project also detects an issue where some `rating` values actually contain duration values such as:

```text
74 min
```

A mask is created to detect these incorrect values:

```python
mask_swapped = df['rating'].astype(str).str.contains('min', na=False)
```

The affected values are then moved to the correct `duration` column and the incorrect rating is replaced with a missing value.

This demonstrates an additional **data validation and correction step** rather than simply filling missing values.

---

## 🗑️ Duplicate Removal

Duplicate records are checked and removed:

```python
df.drop_duplicates(inplace=True)
```

This ensures that duplicated observations do not negatively affect model training.

---

# 📈 Data Visualization

The project uses:

* Matplotlib
* Seaborn

to visualize the dataset.

Examples include:

* Target distribution
* Numeric feature distributions
* Model comparison
* Confusion matrices
* Other exploratory visualizations

A count plot is used to visualize the distribution of:

```text
Movie vs TV Show
```

This helps identify whether the target classes are balanced or imbalanced.

---

# ⚙️ Feature Engineering

The project creates additional numerical and categorical features from the original Netflix columns.

Examples include:

```text
num_cast
num_directors
num_countries
num_genres
main_country
main_genre
```

These engineered features convert useful information from text-based columns into features that Machine Learning algorithms can process.

For example:

```text
cast → num_cast
countries → num_countries
listed_in → num_genres
```

This allows the models to use structural information from the original text fields without directly processing the complete raw text.

---

# 🔤 Encoding

Categorical information is converted into numerical representations so that Machine Learning models can process it.

The project also saves the target encoder:

```text
label_type.pkl
```

This is important because the same encoding used during training must be available when making predictions later.

---

# 🔎 Correlation Analysis

The project performs correlation analysis to identify highly correlated features.

A correlation threshold of:

```text
0.9
```

is used to identify highly correlated features.

This helps reduce redundant information and prevents the model from using unnecessary or highly related features.

The project identifies features that should be excluded because of weak relevance or high correlation.

---

# 🎯 Feature Selection

After feature engineering and correlation analysis, the project creates the final feature set.

The selected features before RFE include engineered and categorical features.

Then **Recursive Feature Elimination (RFE)** is applied.

```python
rfe = RFE(
    estimator=rf_selector,
    n_features_to_select=5,
    step=1
)
```

RFE selects the most useful features recursively.

The final RFE-selected features are:

```text
num_directors
num_genres
rating
main_country
main_genre
```

This reduces the number of features while keeping the most informative ones.

---

# 🔬 Polynomial Features

The project also demonstrates feature expansion using:

```python
PolynomialFeatures(
    degree=2,
    include_bias=False,
    interaction_only=True
)
```

This creates interaction features between variables.

For example:

```text
num_cast
num_genres
```

can produce:

```text
num_cast × num_genres
```

This allows the model to represent relationships between features.

---

# ✂️ Train / Test Split

The dataset is divided into training and testing sets:

```python
x_train, x_test, y_train, y_test = train_test_split(
    x,
    y,
    test_size=0.2,
    random_state=42,
    stratify=y
)
```

The split is:

```text
80% → Training
20% → Testing
```

`random_state=42` makes the split reproducible.

`stratify=y` keeps the class distribution similar between training and testing data.

---

# 📏 Feature Scaling

The project uses:

```python
StandardScaler()
```

to standardize the numerical features.

```python
scaler = StandardScaler()

x_train_scaled = scaler.fit_transform(x_train)
x_test_scaled = scaler.transform(x_test)
```

The scaler is fitted only on the training data and then applied to the test data.

This avoids information leakage from the test set.

The fitted scaler is saved for future predictions:

```text
scaler.pkl
```

---

# 📉 PCA - Dimensionality Reduction

Principal Component Analysis (PCA) is also applied.

```python
pca = PCA(n_components=5)
```

The first five principal components explain approximately:

```text
82.46%
```

of the total variance.

This demonstrates how dimensionality reduction can preserve most of the information while reducing the number of dimensions.

---

# 🤖 Machine Learning Models

Several classification algorithms are trained and evaluated.

## 1. Logistic Regression

```text
Accuracy ≈ 95.63%
```

Logistic Regression is used as a baseline classification model.

It is useful because it is:

* Simple
* Fast
* Interpretable
* Suitable for binary classification

---

## 2. Support Vector Machine (SVM)

```text
Accuracy ≈ 95.57%
```

SVM attempts to find a decision boundary that separates the classes effectively.

---

## 3. K-Nearest Neighbors (KNN)

```text
Accuracy ≈ 96.71%
```

KNN predicts a class based on the nearest training observations.

Because KNN depends on distances, feature scaling is particularly important.

---

## 4. Decision Tree

```text
Accuracy ≈ 99.89%
```

The Decision Tree achieved the highest test accuracy in this project.

It works by creating a sequence of decision rules that split the data into different classes.

The project selected:

```text
Decision Tree
```

as the best-performing model based on the model comparison.

---

## 5. Random Forest

```text
Accuracy ≈ 99.04%
```

Random Forest combines multiple decision trees to produce a stronger ensemble model.

It achieved the second-highest test accuracy in the project.

---

## 6. Artificial Neural Network (ANN)

The project also implements an ANN using **Keras / TensorFlow**.

The neural network contains Dense layers, activation functions, and Dropout.

The final ANN achieved approximately:

```text
96.82% Test Accuracy
```

Dropout is used as a regularization technique to help reduce overfitting.

---

# 📊 Model Comparison

The final test accuracy comparison is:

| Model               | Test Accuracy |
| ------------------- | ------------: |
| 🥇 Decision Tree    |    **99.89%** |
| 🥈 Random Forest    |    **99.04%** |
| ANN (Keras)         |    **96.82%** |
| KNN                 |    **96.71%** |
| Logistic Regression |    **95.63%** |
| SVM                 |    **95.57%** |

The Decision Tree achieved the best test accuracy.

The model comparison is also visualized using a bar chart.

---

# 🔄 Cross Validation

The project uses **5-Fold Cross Validation** to provide a more reliable estimate of model performance.

The results show that Decision Tree also performs very strongly under cross validation:

```text
Decision Tree
CV Mean Accuracy ≈ 99.86%
CV Std ≈ 0.0006
```

This is close to its test accuracy, indicating consistent performance across the validation folds.

The project also compares:

* Test Accuracy
* Cross-Validation Mean Accuracy
* Cross-Validation Standard Deviation

---

# 🏆 Best Model

Based on the model comparison:

```text
Best Model = Decision Tree
```

with:

```text
Test Accuracy ≈ 99.89%
```

The project therefore saves the Decision Tree model for later use.

---

# 💾 Model Saving & Download

The project does not only train the model inside the Notebook.

It also saves the trained model and the preprocessing components so that the model can be reused without retraining.

The generated files include:

```text
best_model.pkl
label_type.pkl
feature_cols.pkl
scaler.pkl
netflix_processed.csv
```

### `best_model.pkl`

Contains the trained best-performing Scikit-learn model.

In this project, the selected model is the:

```text
Decision Tree
```

### `label_type.pkl`

Stores the target-label encoding.

### `feature_cols.pkl`

Stores the feature columns required by the model.

### `scaler.pkl`

Stores the fitted StandardScaler used during preprocessing.

### `netflix_processed.csv`

Contains the processed dataset so that preprocessing does not need to be repeated when testing samples.

The project also contains logic for saving the ANN separately as:

```text
best_model_ann.keras
```

when the ANN is selected as the best model.

---

# 🧪 Model Testing

A separate testing file/notebook is included in the project for testing the saved model.

The purpose of the testing component is to demonstrate that the trained model can be loaded and used independently from the original training notebook.

The testing workflow is:

```text
Load Saved Model
       ↓
Load Saved Encoder
       ↓
Load Feature Columns
       ↓
Load Scaler
       ↓
Prepare Test Data
       ↓
Apply Same Preprocessing
       ↓
Model Prediction
       ↓
Display Prediction
```

This makes the project closer to a real Machine Learning application instead of a model that only works inside the training notebook.

---

# 🌐 Streamlit Application

The project also includes a **Streamlit web application** that provides a user-friendly interface for interacting with the trained Machine Learning model.

Instead of running Python code manually, the user can interact with the model through a web interface.

The Streamlit application loads the saved model and preprocessing files and uses them to generate predictions.

### Streamlit workflow

```text
User Input
    ↓
Streamlit Interface
    ↓
Feature Preparation
    ↓
Saved Scaler / Encoder
    ↓
Saved Decision Tree Model
    ↓
Prediction
    ↓
Movie / TV Show
```

This makes the Machine Learning model accessible through an interactive application.

---

# 🖥️ What the Streamlit App Does

The application provides an interface where the user can enter/select the required information.

The application then:

1. Receives the user's input.
2. Converts the input into the required feature format.
3. Applies the same preprocessing used during training.
4. Loads the saved model.
5. Sends the processed features to the model.
6. Generates a prediction.
7. Displays whether the title is predicted as:

   * 🎬 Movie
   * 📺 TV Show

The important point is that the Streamlit application **does not retrain the model every time**.

Instead, it uses the already-trained model saved from the Machine Learning pipeline.

---

# 📁 Project Structure

A suggested repository structure is:

```text
Netflix-Titles-Type-Prediction/
│
├── Project_NTI_ML.ipynb
│
├── netflix_titles_CLEANED.csv
│
├── netflix_processed.csv
│
├── best_model.pkl
├── label_type.pkl
├── feature_cols.pkl
├── scaler.pkl
│
├── best_model_ann.keras
│
├── Streamlit/
│   └── app.py
│
├── Model_Test/
│   └── model_test.py
│
├── requirements.txt
│
└── README.md
```

> `best_model_ann.keras` is only needed when the Keras ANN model is being saved/used; the current best model selected by the project is the Decision Tree.

---

# 🛠️ Technologies Used

### Programming Language

* Python

### Data Analysis

* Pandas
* NumPy

### Data Visualization

* Matplotlib
* Seaborn

### Machine Learning

* Scikit-learn
* XGBoost

### Deep Learning

* TensorFlow
* Keras

### Deployment

* Streamlit

### Model Serialization

* Pickle

---

# 📚 Main Machine Learning Concepts Covered

This project demonstrates:

* Exploratory Data Analysis
* Data Cleaning
* Missing Value Handling
* Duplicate Detection
* Feature Engineering
* Feature Encoding
* Correlation Analysis
* Feature Selection
* Recursive Feature Elimination (RFE)
* Polynomial Features
* Train/Test Split
* Standardization
* PCA
* Logistic Regression
* SVM
* KNN
* Decision Tree
* Random Forest
* Artificial Neural Networks
* Confusion Matrix
* Classification Report
* Accuracy
* Precision
* Recall
* F1-Score
* Cross Validation
* Model Comparison
* Model Serialization
* Model Testing
* Streamlit Deployment

---

# 🚀 How to Run the Project

## 1. Clone the repository

```bash
git clone <YOUR_GITHUB_REPOSITORY_URL>
cd Netflix-Titles-Type-Prediction
```

## 2. Install the required libraries

```bash
pip install -r requirements.txt
```

## 3. Run the Notebook

Open:

```text
Project_NTI_ML.ipynb
```

and run the cells to reproduce the Machine Learning workflow.

---

## 4. Run the Streamlit Application

From the project directory:

```bash
streamlit run app.py
```

The Streamlit application will open in the browser.

---

# 🎯 Project Outcome

The project successfully builds a complete Machine Learning classification pipeline for Netflix titles.

After preprocessing, feature engineering, feature selection, training, and evaluation, multiple classification algorithms were compared.

The **Decision Tree** achieved the best performance with approximately:

```text
99.89% Test Accuracy
```

and approximately:

```text
99.86% Cross-Validation Mean Accuracy
```

The trained model and its preprocessing components are saved so they can be reused independently.

The project is then extended into a **Streamlit application**, allowing users to interact with the trained model through a simple web interface.

A separate model-testing component is also provided to verify predictions using the saved model.

---

# 👨‍💻 Project Highlights

This project is not limited to training a single Machine Learning model.

It demonstrates the complete process of taking raw data and turning it into a reusable Machine Learning application:

```text
Raw Data
   ↓
Cleaning
   ↓
Analysis
   ↓
Feature Engineering
   ↓
Feature Selection
   ↓
Machine Learning
   ↓
Evaluation
   ↓
Model Selection
   ↓
Model Saving
   ↓
Independent Testing
   ↓
Streamlit Deployment
```

This makes the project suitable as a practical demonstration of **Data Science, Machine Learning, Model Evaluation, Model Deployment, and MLOps-oriented workflow concepts**.
