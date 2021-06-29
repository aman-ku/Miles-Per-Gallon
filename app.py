from flask import Flask, request, render_template
from flask_cors import cross_origin
import sklearn
import pickle
import pandas as pd

app = Flask(__name__)
model = pickle.load(open("MPG_model.pkl", "rb"))

@app.route("/")
@cross_origin()
def home():
    return render_template("home.html")


@app.route("/", methods=['POST'])
@cross_origin()
def predict():
    
    inputQuery1 = request.form['query1']
    inputQuery2 = request.form['query2']
    inputQuery3 = request.form['query3']
    inputQuery4 = request.form['query4']

    prediction=model.predict([[inputQuery1,inputQuery2,inputQuery3,inputQuery4]])

    output = round(prediction[0], 2)
    return render_template('home.html',output1=output)

    
if __name__ == "__main__":
    app.run(debug=True)