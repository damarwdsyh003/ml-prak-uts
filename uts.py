from flask import Flask, render_template, request
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from sklearn.naive_bayes import GaussianNB
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
import numpy as np

app = Flask(__name__)

# Load the preprocessed dataset (dataset assumed to be already cleaned)
file_path = 'Prostate_Cancer_Scaled.csv'
data = pd.read_csv(file_path)

# Ensure 'diagnosis_result' is properly mapped for classification
data['diagnosis_result'] = data['diagnosis_result'].map({'M': 1, 'B': 0})

# Splitting features and target
X = data.drop(['id', 'diagnosis_result'], axis=1)  # Drop 'id' and 'diagnosis_result' from features
y = data['diagnosis_result']  # Target variable

# No need to handle missing values or outliers as data is clean
# Scale the features
scaler = MinMaxScaler()
X_scaled = scaler.fit_transform(X)

# Split the data
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=0)

# Initialize models
nb_model = GaussianNB()
dt_model = DecisionTreeClassifier(random_state=0)
rf_model = RandomForestClassifier(random_state=0)

# Fit models
nb_model.fit(X_train, y_train)
dt_model.fit(X_train, y_train)
rf_model.fit(X_train, y_train)

# Function to predict cancer using input data
def predict_cancer(input_data):
    # Scale the input data
    input_data_scaled = scaler.transform([input_data])
    
    # Make predictions
    nb_pred = nb_model.predict(input_data_scaled)
    dt_pred = dt_model.predict(input_data_scaled)
    rf_pred = rf_model.predict(input_data_scaled)

    # Get predictions in a user-friendly format
    results = {
        "Naive Bayes": 'M (Malignant)' if nb_pred[0] == 1 else 'B (Benign)',
        "Decision Tree": 'M (Malignant)' if dt_pred[0] == 1 else 'B (Benign)',
        "Random Forest": 'M (Malignant)' if rf_pred[0] == 1 else 'B (Benign)'
    }

    # Collect metrics for each model
    metrics = {
        'Naive Bayes': calculate_metrics(y_test, nb_model.predict(X_test)),
        'Decision Tree': calculate_metrics(y_test, dt_model.predict(X_test)),
        'Random Forest': calculate_metrics(y_test, rf_model.predict(X_test))
    }

    confusion_matrices = {
        'Naive Bayes': confusion_matrix(y_test, nb_model.predict(X_test)).tolist(),
        'Decision Tree': confusion_matrix(y_test, dt_model.predict(X_test)).tolist(),
        'Random Forest': confusion_matrix(y_test, rf_model.predict(X_test)).tolist()
    }

    # Get true labels for the last sample in the test set (you may adjust this logic as needed)
    true_labels = {
        "Naive Bayes": 'M (Malignant)' if y_test.iloc[-1] == 1 else 'B (Benign)',
        "Decision Tree": 'M (Malignant)' if y_test.iloc[-1] == 1 else 'B (Benign)',
        "Random Forest": 'M (Malignant)' if y_test.iloc[-1] == 1 else 'B (Benign)',
    }

    return results, metrics, confusion_matrices, true_labels

# Function to calculate metrics
def calculate_metrics(y_true, y_pred):
    metrics = {
        'accuracy': accuracy_score(y_true, y_pred),
        'precision': precision_score(y_true, y_pred, average='weighted'),
        'recall': recall_score(y_true, y_pred, average='weighted'),
        'f1_score': f1_score(y_true, y_pred, average='weighted')
    }
    return metrics

# Route for landing page
@app.route("/")
def landing():
    return render_template("index.html")

# Route for form page
@app.route("/form", methods=["GET", "POST"])
def form():
    if request.method == "POST":
        try:
            # Get input from form
            radius = float(request.form["radius"])
            texture = float(request.form["texture"])
            perimeter = float(request.form["perimeter"])
            area = float(request.form["area"])
            smoothness = float(request.form["smoothness"])
            compactness = float(request.form["compactness"])
            symmetry = float(request.form["symmetry"])
            fractal_dimension = float(request.form["fractal_dimension"])

            # Make prediction
            input_data = [radius, texture, perimeter, area, smoothness, compactness, symmetry, fractal_dimension]
            results, metrics, confusion_matrices, true_labels = predict_cancer(input_data)

            # Redirect to results page with prediction, metrics, confusion matrices, and true labels
            return render_template("results.html", results=results, metrics=metrics, 
            confusion_matrices=confusion_matrices, true_labels=true_labels)

        except Exception as e:
            return f"Error: {str(e)}"
    
    return render_template("form.html")

# Route for results page
@app.route("/results")
def results():
    return render_template("results.html")

if __name__ == "__main__":
    app.run(debug=True)
