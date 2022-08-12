import pandas as pd
from django.shortcuts import render
import joblib
from sklearn.feature_extraction.text import TfidfVectorizer

def home(request):
    return render(request, 'deploymodels/home.html')


def linearregression(request):
    modelresult = None
    model_values = None
    house_area = None
    house_bedrooms = None
    if request.GET.get('area'):
        house_area = request.GET.get("area")
        house_bedrooms = request.GET.get("bedrooms")
        model = joblib.load("finalized_model.sav")
        model_values = [house_area, house_bedrooms]

        modelresult = str(int(model.predict([[int(house_area), int(house_bedrooms)]])))

    return render(request, 'deploymodels/linearregression.html', {"modelresult": modelresult, "house_area": house_area,
                                                                  "house_bedrooms": house_bedrooms})


def logisticregression(request):
    sentence = None
    logistic_result = None

    if request.GET.get("sentence"):
        sentence = request.GET.get("sentence")

        logistic_model = joblib.load("LogisticRegression.sav")
        logistic_vector = joblib.load("LogisticVector.sav")

        predict_val = logistic_vector.transform([sentence])
        logistic_result = logistic_model.predict(predict_val)

        logistic_result_val = str(logistic_result[0])
        if logistic_result_val == "positive":
            logistic_result = "Not Spam"
        else:
            logistic_result = "Spam"

    return render(request, "deploymodels/logisticregression.html", {"logisticmodel": logistic_result})
