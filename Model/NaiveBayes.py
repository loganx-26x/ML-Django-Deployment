import pandas as pd
from sklearn.naive_bayes import CategoricalNB
from sklearn.model_selection import train_test_split
from sklearn.metrics import f1_score
import joblib

data = pd.read_csv('Heart.csv', index_col=0)
features = data.drop(["AHD", "Thal", "Ca", "ChestPain"], axis=1)
labels = data.iloc[:, [-1]]

x_train, Y_train, x_test, Y_test = train_test_split(features, labels, test_size=0.2, random_state=1)

model2 = CategoricalNB()
model2.fit(x_train, x_test.values.ravel())

y_pred = model2.predict(Y_train)
print(f1_score(Y_test, y_pred, pos_label="Yes"))

# save the model to disk
naivebayes_vector = 'NaiveBayes.sav'
joblib.dump(model2, naivebayes_vector)
