import numpy as np
from tensorflow import keras
import joblib

np.random.seed(0)
import tensorflow as tf
tf.random.set_seed(1)

(x_train, y_train), (x_test, y_test) = keras.datasets.mnist.load_data()

x_train_reshaped = x_train.reshape(-1, 28*28)
x_test_reshaped = x_test.reshape(-1, 28*28)

from keras.utils import to_categorical
y_train_cat = to_categorical(y_train, 10)
y_test_cat = to_categorical(y_test, 10)

from keras.models import Sequential
from keras.layers import Dense, Dropout

image_model = Sequential()
image_model.add(Dense(128, activation='relu', input_shape=(28*28,)))
image_model.add(Dropout(.2))
image_model.add(Dense(64, activation='relu'))
image_model.add(Dropout(.2))
image_model.add(Dense(10, activation='softmax'))

image_model.compile(loss='categorical_crossentropy', optimizer="adam", metrics=['accuracy'])
image_model.fit(x_train_reshaped, y_train_cat, epochs=5, batch_size=10)

predictions_vector = image_model.predict(x_test_reshaped)
predictions = [np.argmax(pred) for pred in predictions_vector]

num_correct = 0
for i in range(len(predictions)):
    if predictions[i] == y_test[i]:
        num_correct += 1

print(f"The model is correct {num_correct} times out of {len(y_test)}")
print(f"The accuracy is {num_correct/len(y_test)}")

# save the model to disk
filename = 'SequentialNN.sav'
joblib.dump(image_model, filename)
