from django.shortcuts import render
import joblib


def home(request):
    return render(request, 'deploymodels/home.html')


def linearregression(request):
    return render(request, 'deploymodels/linearregression.html')


def result(request):

    model = joblib.load("finalized_model.sav")

    model_values = [request.GET['price'], request.GET["area"]]

    modelresult=str(int(model.predict([[int(model_values[0]), int(model_values[1])]])))

    return render(request, "deploymodels/result.html", {"modelresult": modelresult, "prediction":model_values})