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

# Load the dataset
file_path = 'Prostate_Cancer_Scaled.csv'
data = pd.read_csv(file_path)

# Pre-processing
data_cleaned = data.dropna()  # Drop rows with missing values
z_scores = np.abs((data_cleaned.select_dtypes(include=[np.number]) -
                   data_cleaned.select_dtypes(include=[np.number]).mean()) / 
                   data_cleaned.select_dtypes(include=[np.number]).std())
data_cleaned = data_cleaned[(z_scores < 3).all(axis=1)]

data_cleaned['diagnosis_result'] = data_cleaned['diagnosis_result'].map({'M': 1, 'B': 0})
X = data_cleaned.drop(['id', 'diagnosis_result'], axis=1)
y = data_cleaned['diagnosis_result']

scaler = MinMaxScaler()
X_scaled = scaler.fit_transform(X)

X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=0)

nb_model = GaussianNB()
dt_model = DecisionTreeClassifier(random_state=0)
rf_model = RandomForestClassifier(random_state=0)

nb_model.fit(X_train, y_train)
dt_model.fit(X_train, y_train)
rf_model.fit(X_train, y_train)

def predict_cancer(input_data):
    input_data_scaled = scaler.transform([input_data])
    
    nb_pred = nb_model.predict(input_data_scaled)
    dt_pred = dt_model.predict(input_data_scaled)
    rf_pred = rf_model.predict(input_data_scaled)
    
    results = {
        "Naive Bayes": 'M (Malignant)' if nb_pred[0] == 1 else 'B (Benign)',
        "Decision Tree": 'M (Malignant)' if dt_pred[0] == 1 else 'B (Benign)',
        "Random Forest": 'M (Malignant)' if rf_pred[0] == 1 else 'B (Benign)'
    }
    return results

@app.route("/", methods=["GET", "POST"])
def index():
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
            results = predict_cancer(input_data)
            
            return render_template("index.html", results=results)
        except Exception as e:
            return f"Error: {str(e)}"
    return render_template("index.html", results=None)

if __name__ == "__main__":
    app.run(debug=True)
