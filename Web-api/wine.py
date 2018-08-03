# Import for Flask
import json
from flask import Flask, redirect
from flask import render_template, request

# import necessary libs for EDA
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

# Import for Machine Learning
from sklearn.metrics import classification_report
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from imblearn.over_sampling import SMOTE

# create the flask object
app = Flask(__name__)

# Web Root - Redirecting to /wine
@app.route('/')
def home():
    return redirect("/wine", code=302)

# Route /win
@app.route('/wine', methods=['GET','POST'])
def wine():

    # Data object to be passed a POST data
    data = {}

    if request.form:

        # get the input data
        form_data = request.form
        json_data = request.form.to_dict()
        data['form'] = form_data

        # get wine properties from form
        fixed_acidity = float(form_data['fixed_acidity'])
        volatile_acidity = float(form_data['volatile_acidity'])
        citric_acid = float(form_data['citric_acid'])
        residual_sugar = float(form_data['residual_sugar'])
        chlorides = float(form_data['chlorides'])
        free_sulfur_dioxide = float(form_data['free_sulfur_dioxide'])
        total_sulfur_dioxide = float(form_data['total_sulfur_dioxide'])
        density = float(form_data['density'])
        sulphates = float(form_data['sulphates'])
        pH = float(form_data['pH'])
        alcohol = float(form_data['alcohol'])

        # Array containing the properties to be predicted
        input_data = np.array([[fixed_acidity, volatile_acidity, citric_acid, residual_sugar,
		                chlorides, free_sulfur_dioxide, total_sulfur_dioxide,
			        density, pH, sulphates, alcohol]])

        # get prediction
        prediction = model.predict(input_data)

        # For debug only - Optional
        print("DEBUG: DATA Pred:", prediction)

        # Save prediction into the data to be submitted as a POST
        data['prediction'] = prediction

        # JSON data
        data['json'] = json.dumps(json_data, sort_keys=True, indent=4)
        print(data['json'])

    # Render Output into HTML (wine.html)
    return render_template('wine.html', data=data)

# Main function
if __name__ == '__main__':

    # Load dataset
    wine = pd.read_csv('./data/winequality-red.csv')
    wine = wine.drop_duplicates(subset=None, keep="first", inplace=False)

    # Set features (X) and target (y)
    X = wine.drop(['quality'], axis=1)
    y = wine['quality']

    # Test Sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=42)

    # SMOTE technique to oversample minority classes due imbalanced data sets
    sample = SMOTE(random_state=0)
    X_smote, y_smote = sample.fit_sample(X, y)

    # Fit Model using Random Forrest
    model = RandomForestClassifier(random_state=42)
    model.fit(X_smote, y_smote)

    # perform predictions on test set
    actual = y_test
    predictions = model.predict(X_test)

    # Print Classificaton Report (to verify the model accuracy)
    print('Classification Report:\n', classification_report(actual, predictions))

    # For testing purposes only - Predicted should be ideally an "8"
    test_pred = model.predict([[7.9, 0.35, 0.46, 3.6, 0.078, 15, 37, 0.9973, 3.35, 0.86, 12.8]])
    print("Prediction:", test_pred)

    # start the app
    app.run(host='0.0.0.0', port='5000')
    app.run(debug = True)

#EOF
