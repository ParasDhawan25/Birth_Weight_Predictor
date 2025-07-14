from flask import Flask, request, jsonify, render_template
import pandas as pd
import pickle
import sklearn


app = Flask(__name__)

def get_cleaned_data(form_data):
    Gestational_Days = int(form_data['Maternal.Gestational.Days'])
    Maternal_age = int(form_data['Maternal.age'])
    Maternal_Height = int(form_data['Maternal.Height'])  
    Maternal_Weight = int(form_data['Maternal.Pregnancy.Weight'])
    Maternal_smoker = int(form_data['Maternal.Smoker'])  

    cleaned_data = {
        "Gestational.Days": [Gestational_Days],
        "Maternal.Age": [Maternal_age],
        "Maternal.Height": [Maternal_Height],
        "Maternal.Pregnancy.Weight": [Maternal_Weight],
        "Maternal.Smoker": [Maternal_smoker]
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