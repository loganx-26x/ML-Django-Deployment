import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
import joblib


movies = pd.read_csv("IMDB_Dataset.csv")

vector = TfidfVectorizer(min_df=1, stop_words="english", lowercase="True")

features = vector.fit_transform(movies["review"])
labels = movies["sentiment"]

X_train, y_train, X_test, y_test = train_test_split(features, labels, test_size=0.2, random_state=1)

log_reg = LogisticRegression(solver='saga')
log_reg.fit(X_train, X_test)

prediction = log_reg.predict(y_train)
print(f1_score(prediction, y_test, pos_label="positive"))


# save the model to disk
filename = 'LogisticRegression.sav'
logistic_vector = "LogisticVector.sav"
joblib.dump(log_reg, filename)
joblib.dump(vector, logistic_vector)
