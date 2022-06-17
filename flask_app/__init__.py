from flask import Flask, render_template, request
import pickle
import numpy as np
import pandas as pd
import joblib

# model = pickle.load(open('model/salary_model.pkl', 'rb'))

# app = Flask(__name__)

# @app.route('/')
# def main():
#     return render_template('home.html')

# @app.route('/predict', methods=['POST'])
# def home():
#     data1 = request.form['a']
#     data2 = request.form['b']
#     arr = np.array([[data1, data2]])
#     pred = model.predict(arr)
#     return render_template('after.html', data=pred)

# if __name__ == "__main__":
#     app.run(debug=True)

# Declare a Flask app
app = Flask(__name__)

# Main function here
# ------------------

# Running the app
if __name__ == '__main__':
    app.run(debug = True)


@app.route('/', methods=['GET', 'POST'])
def main():
    
    # If a form is submitted
    if request.method == "POST":
        
        # Unpickle classifier
        salary_model = joblib.load("C:/Users/JEON_SANGEON/codestates/project/toy/project_land/flask_app/salary_model.pkl")
        
        # Get values through input bars
        면적 = request.form.get("면적")
        
        
        # Put inputs to dataframe
        X = pd.DataFrame([[면적]], columns = ["면적"])
        
        # Get prediction
        prediction = salary_model.predict(X)[0]
        
    else:
        prediction = ""
        
    return render_template("home.html", output = prediction)