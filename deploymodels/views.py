from django.shortcuts import render
import joblib


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


def result(request):
    model = joblib.load("finalized_model.sav")

    price = request.GET['price']
    area = request.GET["area"]

    modelresult = str(int(model.predict([[int(price), int(model_values[1])]])))

    return render(request, "deploymodels/result.html", {"modelresult": modelresult, "prediction": model_values})
