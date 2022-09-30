import pickle
from flask import Flask,request,app,url_for,render_template
import numpy as np
import pandas as pd

app = Flask(__name__)
#Load the model
model = pickle.load(open('model.pkl','rb'))

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict',methods=['POST'])
def predict():
    data = [int(x) for x in request.form.values()] # whatever we get data, it is stored in data
    new_data = np.array(data).reshape(1,-1)
    output = model.predict(new_data)[0]
    return render_template('index.html',prediction_text = f"The mobile comes in {output} class")

if __name__ == '__main__':
    app.run(debug=True)
