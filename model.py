import tkinter as tk
from tkinter import messagebox
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, precision_score
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# Load the data
data = pd.read_excel("FoodList1.xlsx")
food_description = data["Food"]
vectorizer = TfidfVectorizer()
features_vectorized = vectorizer.fit_transform(food_description)
target_cuisine = data["Cuisine"]
target_dietary = data["Dietary Preference"]
target_meal_course = data["Meal Course"]
target_prepare = data["Preparation Method"]
target_main = data["Main Ingredient"]

# Train models
X_train, X_test, y_train_cuisine, y_test_cuisine = train_test_split(features_vectorized, target_cuisine, test_size=0.2, random_state=42)
X_train, X_test, y_train_dietary, y_test_dietary = train_test_split(features_vectorized, target_dietary, test_size=0.2, random_state=42)
X_train, X_test, y_train_meal_course, y_test_meal_course = train_test_split(features_vectorized, target_meal_course, test_size=0.2, random_state=42)
X_train, X_test, y_train_prepare, y_test_prepare = train_test_split(features_vectorized, target_prepare, test_size=0.2, random_state=42)
X_train, X_test, y_train_main, y_test_main = train_test_split(features_vectorized, target_main, test_size=0.2, random_state=42)

model_cuisine = RandomForestClassifier()
model_cuisine.fit(X_train, y_train_cuisine)

model_prepare = RandomForestClassifier()
model_prepare.fit(X_train, y_train_prepare)

model_dietary = RandomForestClassifier()
model_dietary.fit(X_train, y_train_dietary)

model_meal_course = RandomForestClassifier()
model_meal_course.fit(X_train, y_train_meal_course)

model_main = RandomForestClassifier()
model_main.fit(X_train, y_train_main)

def predict_food_details(food_item):
    user_features = vectorizer.transform([food_item.lower()])
    predicted_cuisine = model_cuisine.predict(user_features)[0]
    predicted_dietary = model_dietary.predict(user_features)[0]
    predicted_meal_course = model_meal_course.predict(user_features)[0]
    predicted_prepare = model_prepare.predict(user_features)[0]
    predicted_main = model_main.predict(user_features)[0]
    return {
        "Food": food_item,
        "Cuisine": predicted_cuisine,
        "Dietary Preference": predicted_dietary,
        "Preparation Method": predicted_prepare,
        "Meal Course": predicted_meal_course,
        "Main Ingredient": predicted_main + " based"
    }

def show_prediction():
    food_item = food_entry.get()
    if not food_item:
        messagebox.showwarning("Input Error", "Please enter a food item.")
        return
    
    predictions = predict_food_details(food_item)
    
    result_text = f"Predictions for '{food_item}':\n"
    for key, value in predictions.items():
        result_text += f"  {key}: {value}\n"
    
    result_label.config(text=result_text)

def show_metrics():
    cuisine_accuracy = accuracy_score(y_test_cuisine, model_cuisine.predict(X_test))
    dietary_accuracy = accuracy_score(y_test_dietary, model_dietary.predict(X_test))
    meal_course_accuracy = accuracy_score(y_test_meal_course, model_meal_course.predict(X_test))
    prepare_accuracy = accuracy_score(y_test_prepare, model_prepare.predict(X_test))
    
    cuisine_precision = precision_score(y_test_cuisine, model_cuisine.predict(X_test), average='weighted', zero_division='warn')
    dietary_precision = precision_score(y_test_dietary, model_dietary.predict(X_test), average='weighted', zero_division='warn')
    meal_course_precision = precision_score(y_test_meal_course, model_meal_course.predict(X_test), average='weighted', zero_division='warn')
    prepare_precision = precision_score(y_test_prepare, model_prepare.predict(X_test), average='weighted', zero_division='warn')

    metrics_text = f"\nAccuracy:\n  Cuisine: {cuisine_accuracy:.4f}\n  Dietary: {dietary_accuracy:.4f}\n"
    metrics_text += f"  Meal Course: {meal_course_accuracy:.4f}\n  Preparation: {prepare_accuracy:.4f}\n"
    metrics_text += f"\nPrecision:\n  Cuisine: {cuisine_precision:.4f}\n  Dietary: {dietary_precision:.4f}\n"
    metrics_text += f"  Meal Course: {meal_course_precision:.4f}\n  Preparation: {prepare_precision:.4f}\n"
    
    metrics_label.config(text=metrics_text)

def show_bar_chart():
    accuracy_scores = [
        accuracy_score(y_test_cuisine, model_cuisine.predict(X_test)),
        accuracy_score(y_test_dietary, model_dietary.predict(X_test)),
        accuracy_score(y_test_meal_course, model_meal_course.predict(X_test)),
        accuracy_score(y_test_prepare, model_prepare.predict(X_test))
    ]
    precision_scores = [
        precision_score(y_test_cuisine, model_cuisine.predict(X_test), average='weighted', zero_division='warn'),
        precision_score(y_test_dietary, model_dietary.predict(X_test), average='weighted', zero_division='warn'),
        precision_score(y_test_meal_course, model_meal_course.predict(X_test), average='weighted', zero_division='warn'),
        precision_score(y_test_prepare, model_prepare.predict(X_test), average='weighted', zero_division='warn')
    ]
    
    categories = ['Cuisine', 'Dietary', 'Meal Course', 'Preparation']
    fig, ax = plt.subplots()
    bar_width = 0.35
    index = np.arange(len(categories))
    
    accuracy_bar = plt.bar(index, accuracy_scores, bar_width, color='b', label='Accuracy')
    precision_bar = plt.bar(index + bar_width, precision_scores, bar_width, color='g', label='Precision')

    plt.xlabel('Category')
    plt.ylabel('Scores')
    plt.title('Accuracy and Precision by Category')
    plt.xticks(index + bar_width / 2, categories)
    plt.legend()

    plt.tight_layout()
    plt.show()

# Set up the UI
root = tk.Tk()
root.title("Food Prediction App")

# Input section
food_label = tk.Label(root, text="Enter Food Item:")
food_label.grid(row=0, column=0, padx=10, pady=10)
food_entry = tk.Entry(root)
food_entry.grid(row=0, column=1, padx=10, pady=10)

# Buttons
predict_button = tk.Button(root, text="Predict", command=show_prediction)
predict_button.grid(row=0, column=2, padx=10, pady=10)

metrics_button = tk.Button(root, text="Show Metrics", command=show_metrics)
metrics_button.grid(row=1, column=0, columnspan=2, padx=10, pady=10)

chart_button = tk.Button(root, text="Show Bar Chart", command=show_bar_chart)
chart_button.grid(row=1, column=2, padx=10, pady=10)

# Results and metrics display
result_label = tk.Label(root, text="", justify="left")
result_label.grid(row=2, column=0, columnspan=3, padx=10, pady=10)

metrics_label = tk.Label(root, text="", justify="left")
metrics_label.grid(row=3, column=0, columnspan=3, padx=10, pady=10)

root.mainloop()
