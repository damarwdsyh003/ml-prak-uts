import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from sklearn.naive_bayes import GaussianNB
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
import numpy as np

# Load the dataset
file_path = 'prostate_cancerone.csv'
data = pd.read_csv(file_path)

# Pre-processing

# Check for missing values
data_cleaned = data.dropna()  # Drop rows with missing values (if any)

# Outlier detection: using Z-score to detect outliers
z_scores = np.abs((data_cleaned.select_dtypes(include=[np.number]) - 
                   data_cleaned.select_dtypes(include=[np.number]).mean()) / 
                   data_cleaned.select_dtypes(include=[np.number]).std())

# Remove outliers: Removing rows where any value has a z-score > 3
data_cleaned = data_cleaned[(z_scores < 3).all(axis=1)]

# Encode the target variable (diagnosis_result) 'M' -> 1 (Malignant), 'B' -> 0 (Benign)
data_cleaned['diagnosis_result'] = data_cleaned['diagnosis_result'].map({'M': 1, 'B': 0})

# Splitting data into features and target
X = data_cleaned.drop(['id', 'diagnosis_result'], axis=1)
y = data_cleaned['diagnosis_result']

# Normalize the features using Min-Max scaling
scaler = MinMaxScaler()
X_scaled = scaler.fit_transform(X)

# Split data into training and testing sets (80% train, 20% test)
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=0)

# Model Training and Evaluation

# Initialize the models
nb_model = GaussianNB()
dt_model = DecisionTreeClassifier(random_state=0)
rf_model = RandomForestClassifier(random_state=0)

# Train the models
nb_model.fit(X_train, y_train)
dt_model.fit(X_train, y_train)
rf_model.fit(X_train, y_train)

# Make predictions on the test set
y_pred_nb = nb_model.predict(X_test)
y_pred_dt = dt_model.predict(X_test)
y_pred_rf = rf_model.predict(X_test)

# Function to evaluate model performance
def evaluate_model(y_true, y_pred):
    return {
        "Accuracy": accuracy_score(y_true, y_pred),
        "Precision": precision_score(y_true, y_pred),
        "Recall": recall_score(y_true, y_pred),
        "F1 Score": f1_score(y_true, y_pred),
        "Confusion Matrix": confusion_matrix(y_true, y_pred)
    }

# Evaluation results for each model
nb_results = evaluate_model(y_test, y_pred_nb)
dt_results = evaluate_model(y_test, y_pred_dt)
rf_results = evaluate_model(y_test, y_pred_rf)

# Output the results
print("Naive Bayes Results:", nb_results)
print("Decision Tree Results:", dt_results)
print("Random Forest Results:", rf_results)

# Function for user input prediction
def predict_cancer(radius, texture, perimeter, area, smoothness, compactness, symmetry, fractal_dimension):
    # Create a DataFrame with the input values
    input_data = pd.DataFrame([[radius, texture, perimeter, area, smoothness, compactness, symmetry, fractal_dimension]], 
                              columns=['radius', 'texture', 'perimeter', 'area', 'smoothness', 'compactness', 'symmetry', 'fractal_dimension'])
    
    # Normalize the input data using the same scaler as the training data
    input_data_scaled = scaler.transform(input_data)
    
    # Predict using the three models
    nb_pred = nb_model.predict(input_data_scaled)
    dt_pred = dt_model.predict(input_data_scaled)
    rf_pred = rf_model.predict(input_data_scaled)
    
    # Output the predictions
    print("Naive Bayes Prediction:", 'M (Malignant)' if nb_pred[0] == 1 else 'B (Benign)')
    print("Decision Tree Prediction:", 'M (Malignant)' if dt_pred[0] == 1 else 'B (Benign)')
    print("Random Forest Prediction:", 'M (Malignant)' if rf_pred[0] == 1 else 'B (Benign)')

# Get input from the user
radius = float(input("Enter radius: "))
texture = float(input("Enter texture: "))
perimeter = float(input("Enter perimeter: "))
area = float(input("Enter area: "))
smoothness = float(input("Enter smoothness: "))
compactness = float(input("Enter compactness: "))
symmetry = float(input("Enter symmetry: "))
fractal_dimension = float(input("Enter fractal_dimension: "))

# Make prediction based on user input
predict_cancer(radius, texture, perimeter, area, smoothness, compactness, symmetry, fractal_dimension)
