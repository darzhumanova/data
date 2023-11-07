import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.metrics import mean_squared_error
from sklearn.metrics import accuracy_score


def load_data(file_path):
    data = pd.read_csv(file_path)
    return data


def preprocess_data(data):
    data = pd.get_dummies(data, columns=['City'], drop_first=True)
    data = pd.get_dummies(data, columns=['Gender'], drop_first=True)

    X = data[
        ['Latitude', 'Longitude', 'Unemployment Rate']]
    y = data['Inflation Rate']
    return X, y

def train_model(X, y):
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    model = GradientBoostingRegressor(n_estimators=100, verbose=1)
    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)

    mse = mean_squared_error(y_test, y_pred)
    print(f'MSE: {mse}')

    mse_values = []
    for y_pred_epoch in model.staged_predict(X_test):
        mse_epoch = mean_squared_error(y_test, y_pred_epoch)
        mse_values.append(mse_epoch)

    plt.figure(figsize=(10, 6))
    plt.plot(range(1, len(mse_values) + 1), mse_values, marker='o', linestyle='--', color='b')
    plt.title('MSE по эпохам')
    plt.xlabel('Эпохи')
    plt.ylabel('MSE')
    plt.show()

    return mse


def plot_scatter(data):
    plt.figure(figsize=(10, 6))
    sns.scatterplot(x='Longitude', y='Latitude', data=data, hue='City')
    plt.title('Scatter Plot of Latitude and Longitude')
    plt.show()


def plot_income_distribution(data):
    plt.figure(figsize=(10, 6))
    sns.histplot(data['Inflation Rate'], bins=20, kde=True)
    plt.title('Инфляция')
    plt.show()


def plot_education_distribution(data, education_cols):
    plt.figure(figsize=(10, 6))
    for col in education_cols:
        sns.histplot(data[col], bins=20, kde=True, label=col)
    plt.title('Образование')
    plt.legend()
    plt.show()


def plot_unemployment_rate(data):
    plt.figure(figsize=(10, 6))
    sns.histplot(data['Unemployment Rate'], bins=20, kde=True)
    plt.title('Безработица')
    plt.show()


if _name_ == "_main_":
    data = load_data('education_kz_dataset.csv')
    X, y = preprocess_data(data)
    mse = train_model(X, y)
    print(f'MSE: {mse}')

    plot_scatter(data)
    plot_income_distribution(data)

    education_cols = ['High School (%)', 'Bachelor (%)', 'Master (%)', 'PhD (%)']
    plot_education_distribution(data, education_cols)
    plot_unemployment_rate(data)
