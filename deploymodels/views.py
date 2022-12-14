import pandas as pd
from django.shortcuts import render
import joblib
from sklearn.feature_extraction.text import TfidfVectorizer
from PIL import Image
from .forms import ImageForm
import cv2
from .models import Image


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


def neural_network(request):
    """
    import numpy as np
    image_cls = None
    image_val = None

    if request.GET.get("image"):
        with Image.open(request.GET.get("image")) as im:
            image_val = im.load()
        image_classifier = joblib.load("./LogisticRegression.sav")
        image_cls = image_classifier.predict(image_val)
        # predictions = [np.argmax(pred) for pred in image_cls]

    return render(request, "deploymodels/neuralnetwork.html", {"image_cls": image_cls})
    """
    """Process images uploaded by users"""
    import numpy as np
    if request.method == 'POST':
        form = ImageForm(request.POST, request.FILES)
        if form.is_valid():
            form.save()
            # Get the current instance object to display in the template
            img_obj = form.instance
            image = cv2.imread(Image.objects.all(), cv2.COLOR_BGR2GRAY)
            print(Image.objects.all())
            """new_shaped_image = np.reshape(image, (28, 28))

            image_classifier = joblib.load("./LogisticRegression.sav")
            image_cls = image_classifier.predict(new_shaped_image)"""
            return render(request, 'deploymodels/neuralnetwork.html', {'form': form, 'img_obj': img_obj})
    else:
        form = ImageForm()
    return render(request, 'deploymodels/neuralnetwork.html', {'form': form})


def naive_bayes(request):
    naivebayes_result = None
    if request.GET.get("age"):
        age = request.GET.get("age")
        gender = request.GET.get("gender")
        trestbps = request.GET.get("tbps")
        chol = request.GET.get("chol")
        fbs = request.GET.get("fbs")
        restecg = request.GET.get("restecg")
        tha = request.GET.get("tha")
        exang = request.GET.get("exang")
        olpe = request.GET.get("olpe")
        slope = request.GET.get("slope")

        naivebayes_model = joblib.load("NaiveBayes.sav")
        model_values = [age, gender, trestbps, chol, fbs, restecg, tha, exang, olpe, slope]

        modelresult = naivebayes_model.predict([[int(age), int(gender), int(trestbps), int(chol), int(fbs),
                                                 int(restecg), int(tha), int(exang), int(olpe), int(slope)]])

        if modelresult == "No":
            naivebayes_result = "Safe"
        else:
            naivebayes_result = "Heart Problems"

    return render(request, 'deploymodels/naivebayesian.html', {"result": naivebayes_result})
