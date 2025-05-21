import pickle
from flask import Flask, request, render_template
import numpy as np
from sklearn.preprocessing import LabelEncoder

app = Flask(__name__, template_folder='.')

# Load the pre-trained model and LabelEncoders
with open('heart.pkl', 'rb') as file:
    model = pickle.load(file)

# Manually encode the label categories for consistent encoding
categories = {
    'general_health': ['poor', 'fair', 'good', 'very good', 'excellent'],
    'exercise': ['no', 'yes'],
    'diabetes': ['no', 'yes'],
    'sex': ['male', 'female'],
    'smoking_history': ['no', 'yes'],  # Corrected for binary Yes/No
    'alcohol_consumption': ['no', 'yes']  # Corrected for binary Yes/No
}

# Create label encoders for each category to prevent unseen labels
encoders = {
    key: LabelEncoder().fit(value) for key, value in categories.items()
}


@app.route('/index1')
def index():
    return render_template('index1.html')


@app.route('/predict', methods=['POST'])
def predict():
    data = request.form

    # Use the fitted encoders to transform the data
    general_health = encoders['general_health'].transform(
        [data['general_health'].lower()]
    )[0]
    exercise = encoders['exercise'].transform([data['exercise'].lower()])[0]
    diabetes = encoders['diabetes'].transform([data['diabetes'].lower()])[0]
    sex = encoders['sex'].transform([data['sex'].lower()])[0]
    smoking_history = encoders['smoking_history'].transform(
        [data['smoking_history'].lower()]
    )[0]
    alcohol_consumption = encoders['alcohol_consumption'].transform(
        [data['alcohol_consumption'].lower()]
    )[0]

    # Create input with only the selected features (reshaped as 2D array)
    inputs = np.array([[general_health, exercise, diabetes, sex,
                        smoking_history, alcohol_consumption]])

    # Make prediction
    prediction = model.predict(inputs)[0]
    result = "Yes" if prediction == 1 else "No"

    return render_template('result1.html', result=result)


if __name__ == '__main__':
    app.run(debug=True)
