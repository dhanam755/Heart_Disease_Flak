from flask import Flask, render_template, request
import pickle
import pandas as pd

model = pickle.load(open('model.pkl', 'rb'))
model_columns = pickle.load(open('columns.pkl', 'rb'))  

app = Flask(__name__)

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    # Collect form inputs
    input_data = {
        'Age': int(request.form['Age']),
        'Sex': request.form['Sex'],
        'ChestPainType': request.form['ChestPainType'],
        'RestingBP': int(request.form['RestingBP']),
        'Cholesterol': int(request.form['Cholesterol']),
        'FastingBS': int(request.form['FastingBS']),
        'RestingECG': request.form['RestingECG'],
        'MaxHR': int(request.form['MaxHR']),
        'ExerciseAngina': request.form['ExerciseAngina'],
        'Oldpeak': float(request.form['Oldpeak']),
        'ST_Slope': request.form['ST_Slope']
    }

    input_df = pd.DataFrame([input_data])

    input_encoded = pd.get_dummies(input_df)

    input_encoded = input_encoded.reindex(columns=model_columns, fill_value=0)

    prediction = model.predict(input_encoded)[0]
    result = "Heart Disease Possible" if prediction == 1 else "Heart Disease Not Likely"

    return render_template('result.html', prediction=result)

if __name__ == '__main__':
    app.run(debug=True)
