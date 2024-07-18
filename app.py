import pickle
from flask import Flask, render_template, request, app, jsonify 
import numpy as np
import pandas as pd


app = Flask(__name__)
model = pickle.load(open('notebook/best_model.pkl', 'rb'))

@app.route('/')
def home():
    return render_template('home.html')

@app.route('/predict_api', methods=['POST'])
def predict_api():
    data = request.json['data']
    print(data)
    prediction = model.predict(np.array(list(data.values())).reshape(1, -1))
    output = prediction[0]
    return str(output)

@app.route('/predict', methods=['POST'])
def predict():
    if request.method == 'POST':
        data = {
            "age": int(request.form['age']),
            "sex": bool(request.form['sex']),
            "on_thyroxine": bool(request.form['on_thyroxine']),
            "query_on_thyroxine": bool(request.form['query_on_thyroxine']),
            "on_antithyroid_medication": bool(request.form['on_antithyroid_medication']),
            "sick": bool(request.form['sick']),
            "pregnant": bool(request.form['pregnant']),
            "thyroid_surgery": bool(request.form['thyroid_surgery']),
            "I131_treatment": bool(request.form['I131_treatment']),
            "query_hypothyroid": bool(request.form['query_hypothyroid']),
            "query_hyperthyroid": bool(request.form['query_hyperthyroid']),
            "lithium": bool(request.form['lithium']),
            "goitre": bool(request.form['goitre']),
            "tumor": bool(request.form['tumor']),
            "hypopituitary": bool(request.form['hypopituitary']),
            "psych": bool(request.form['psych']),
            "TSH_measured": bool(request.form['TSH_measured']),
            "TSH": float(request.form['TSH']),
            "T3_measured": bool(request.form['T3_measured']),
            "T3": float(request.form['T3']),
            "TT4_measured": bool(request.form['TT4_measured']),
            "TT4": float(request.form['TT4']),
            "T4U_measured": bool(request.form['T4U_measured']),
            "T4U": float(request.form['T4U']),
            "FTI_measured": bool(request.form['FTI_measured']),
            "FTI": float(request.form['FTI'])
        }
        
        # Convert to DataFrame
        data_df = pd.DataFrame([data])
        
        # Ensure the order of columns matches the training data
        feature_columns = ['age', 'sex', 'on_thyroxine', 'query_on_thyroxine',
                           'on_antithyroid_medication', 'sick', 'pregnant', 'thyroid_surgery',
                           'I131_treatment', 'query_hypothyroid', 'query_hyperthyroid', 'lithium',
                           'goitre', 'tumor', 'hypopituitary', 'psych', 'TSH_measured', 'TSH',
                           'T3_measured', 'T3', 'TT4_measured', 'TT4', 'T4U_measured', 'T4U',
                           'FTI_measured', 'FTI']
        
        data_df = data_df[feature_columns]
        
        # Make prediction
        prediction = model.predict(data_df)
        output = prediction[0]
        
        # Suggestions and hospitals
        suggestions = ""
        if output == 1:
            suggestions = {
                "message": "You may have a thyroid disease.",
                "tips": [
                    "Eat a balanced diet rich in fruits and vegetables.",
                    "Avoid processed foods and sugary snacks.",
                    "Take prescribed medication regularly.",
                    "Exercise regularly."
                ],
                "hospitals": [
                    "All India Institute of Medical Sciences (AIIMS), New Delhi",
                    "Christian Medical College (CMC), Vellore",
                    "Tata Memorial Hospital, Mumbai",
                    "Fortis Memorial Research Institute, Gurgaon",
                    "Apollo Hospitals, Chennai"
                ]
            }
        else:
            suggestions = {
                "message": "You do not have a thyroid disease.",
                "tips": [
                "Maintain a healthy lifestyle to prevent thyroid issues.",
                "Regular check-ups can help detect thyroid problems early.",
                "Eat a diet that includes iodine-rich foods like fish, dairy, and iodized salt."
            ],
            "hospitals": []
        }
    
    return render_template('result.html', prediction=output, suggestions=suggestions)



if __name__=="__main__":
    app.run(debug=True)