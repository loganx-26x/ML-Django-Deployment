from sklearn.linear_model import LinearRegression
import math
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
import joblib

data = pd.read_csv("Hyderabad.csv")
features = data[["Area", "No. of Bedrooms"]]
labels = data[["Price"]]

X_train, y_train, X_test, y_test = train_test_split(features, labels, test_size=0.2, random_state=1)
model = LinearRegression()
model.fit(X_train, X_test)

y_true = model.predict(y_train)

print(math.sqrt(mean_squared_error(y_true, y_test)))

# save the model to disk
filename = 'finalized_model.sav'
joblib.dump(model, filename)
