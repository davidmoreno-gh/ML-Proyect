# functions.py

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split, GridSearchCV, cross_val_score
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
from imblearn.over_sampling import SMOTE
from imblearn.under_sampling import RandomUnderSampler
from imblearn.pipeline import Pipeline as imbPipeline

def load_data(filepath):
    """Load data from a CSV file."""
    return pd.read_csv(filepath)

def remove_duplicates(df):
    """Remove duplicate rows from the DataFrame."""
    return df.drop_duplicates()

def count_distinct_values(df):
    """Print the number of distinct values for each column in the DataFrame."""
    for column in df.columns:
        num_distinct_values = len(df[column].unique())
        print(f"{column}: {num_distinct_values} distinct values")

def count_null_values(df):
    """Print the number of null values for each column in the DataFrame."""
    print(df.isnull().sum())

def remove_unnecessary_values(df):
    """Remove rows where the 'gender' column has the value 'Other'."""
    return df[df['gender'] != 'Other']

def plot_histogram(df, column, bins=30):
    """Plot a histogram for a specified column in the DataFrame."""
    plt.hist(df[column], bins=bins, edgecolor='black')
    plt.title(f'{column} Distribution')
    plt.xlabel(column)
    plt.ylabel('Count')
    plt.show()

def plot_countplot(df, column):
    """Plot a countplot for a specified column in the DataFrame."""
    sns.countplot(x=column, data=df)
    plt.title(f'{column} Distribution')
    plt.show()

def plot_distplot(df, column, bins=30):
    """Plot a distribution plot for a specified column in the DataFrame."""
    sns.distplot(df[column], bins=bins)
    plt.title(f'{column} Distribution')
    plt.show()

def plot_boxplot(df, x_col, y_col):
    """Plot a boxplot for specified columns in the DataFrame."""
    sns.boxplot(x=x_col, y=y_col, data=df)
    plt.title(f'{y_col} vs {x_col}')
    plt.show()

def plot_pairplot(df, hue):
    """Plot a pairplot for the DataFrame with a specified hue."""
    sns.pairplot(df, hue=hue)
    plt.show()

def plot_scatterplot(df, x_col, y_col, hue):
    """Plot a scatterplot for specified columns in the DataFrame."""
    sns.scatterplot(x=x_col, y=y_col, hue=hue, data=df)
    plt.title(f'{x_col} vs {y_col}')
    plt.show()

def plot_violinplot(df, x_col, y_col, hue):
    """Plot a violin plot for specified columns in the DataFrame."""
    sns.violinplot(x=x_col, y=y_col, hue=hue, split=True, data=df)
    plt.title(f'{y_col} vs {x_col} split by {hue}')
    plt.show()

def recategorize_smoking(smoking_status):
    """Recategorize smoking status."""
    if smoking_status in ['never', 'No Info']:
        return 'non-smoker'
    elif smoking_status == 'current':
        return 'current'
    elif smoking_status in ['ever', 'former', 'not current']:
        return 'past_smoker'

def perform_one_hot_encoding(df, column_name):
    """Perform one-hot encoding on a specified column in the DataFrame."""
    dummies = pd.get_dummies(df[column_name], prefix=column_name)
    df = pd.concat([df.drop(column_name, axis=1), dummies], axis=1)
    return df

def plot_heatmap(correlation_matrix, title):
    """Plot a heatmap for the correlation matrix."""
    plt.figure(figsize=(15, 10))
    sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', linewidths=0.5, fmt='.2f')
    plt.title(title)
    plt.show()

def train_knn(X_train, y_train, n_neighbors=3):
    """Train a K-Nearest Neighbors classifier."""
    knn = KNeighborsClassifier(n_neighbors=n_neighbors)
    knn.fit(X_train, y_train)
    return knn

def evaluate_model(model, X_test, y_test):
    """Evaluate the model using the test data."""
    y_pred = model.predict(X_test)
    print("Confusion Matrix:")
    print(confusion_matrix(y_test, y_pred))
    print("\nClassification Report:")
    print(classification_report(y_test, y_pred))

def train_logistic_regression(X_train, y_train):
    """Train a Logistic Regression model."""
    log = LogisticRegression()
    log.fit(X_train, y_train)
    return log

def train_random_forest(X_train, y_train, param_grid):
    """Train a Random Forest classifier with GridSearchCV."""
    numeric_features = ['age', 'bmi', 'HbA1c_level', 'blood_glucose_level', 'hypertension', 'heart_disease']
    categorical_features = ['gender', 'smoking_history']
    
    missing_numeric = [col for col in numeric_features if col not in X_train.columns]
    missing_categorical = [col for col in categorical_features if col not in X_train.columns]
    
    if missing_numeric or missing_categorical:
        raise ValueError(f"Missing columns in X_train: {missing_numeric + missing_categorical}")
    
    preprocessor = ColumnTransformer(
        transformers=[
            ('num', StandardScaler(), numeric_features),
            ('cat', OneHotEncoder(), categorical_features)
        ])
    
    over = SMOTE(sampling_strategy=0.1)
    under = RandomUnderSampler(sampling_strategy=0.5)
    clf = imbPipeline(steps=[('preprocessor', preprocessor),
                             ('over', over),
                             ('under', under),
                             ('classifier', RandomForestClassifier())])
    
    grid_search = GridSearchCV(clf, param_grid, cv=5, error_score='raise')
    grid_search.fit(X_train, y_train)
    
    return grid_search

def plot_feature_importances(grid_search, feature_names):
    """Plot feature importances from the best estimator of GridSearchCV."""
    importances = grid_search.best_estimator_.named_steps['classifier'].feature_importances_
    importance_df = pd.DataFrame({'Feature': feature_names, 'Importance': importances})
    importance_df = importance_df.sort_values('Importance', ascending=False)
    print(importance_df)
    plt.figure(figsize=(12, 8))
    sns.barplot(x='Importance', y='Feature', data=importance_df)
    plt.title('Feature Importances')
    plt.show()

def perform_cross_validation(model, X, y, cv=5):
    """Perform cross-validation and print the scores."""
    cv_scores = cross_val_score(model, X, y, cv=cv, scoring='accuracy')
    print(f"Cross-Validation Scores: {cv_scores}")
    print(f"Mean CV Score: {cv_scores.mean()}")
    print(f"Standard Deviation of CV Score: {cv_scores.std()}")

def perform_grid_search(model, param_grid, X_train, y_train, cv=5):
    """Perform GridSearchCV and return the best estimator."""
    grid_search = GridSearchCV(model, param_grid, cv=cv, scoring='accuracy')
    grid_search.fit(X_train, y_train)
    print("Best Parameters: ", grid_search.best_params_)
    return grid_search