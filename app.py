from flask import Flask, request, jsonify, render_template
import pandas as pd
import pickle
import sklearn


app = Flask(__name__)

def get_cleaned_data(form_data):
    gestation = (form_data['Maternal.Gestational.Days'])
    parity = int(form_data['Parity'])
    age = int(form_data['Maternal.age'])
    height = int(form_data['Maternal.Height'])  
    weight = int(form_data['Maternal.Pregnancy.Weight'])
    smoke = int(form_data['Maternal.Smoker'])  

    cleaned_data = {
        "gestation": [gestation],
        "parity": [parity],
        "age": [age],
        "height": [height],
        "weight": [weight],
        "smoke": [smoke]
    }

    return cleaned_data




@app.route("/",methods=['GET'])
def Home_Page():
    return render_template("index.html")


## define your endpoint
@app.route("/predict", methods = ['POST'])
def get_prediction():
    # get data from user
    baby_data = request.form

    baby_data_cleaned=get_cleaned_data(baby_data)
    # convert into dataframe
    baby_df = pd.DataFrame(baby_data_cleaned)

    # load machine learning trained model
    with open("model.pkl", 'rb') as obj:
        model = pickle.load(obj)

    # make prediction on user data
    prediction = model.predict(baby_df)
    prediction = round(float(prediction), 2)

    # return response in a json format
    response = {"Prediction": prediction}

    return render_template("index.html", prediction=prediction)



if __name__=='__main__':
    app.run(debug=True)