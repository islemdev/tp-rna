import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPClassifier
from sklearn import metrics
from datetime import datetime
from pretty_confusion_matrix import pp_matrix_from_data
import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from keras.layers import Activation, Dense,SimpleRNN


#q1
df = pd.read_csv('Iris.csv')
#q2
print(df.head(10))

#q3
print(df.shape)

#q4
sns.pairplot(data=df, vars={"SepalLengthCm", "SepalWidthCm", "PetalLengthCm", "PetalWidthCm"}, hue='Species')


plt.show()

#q5
df['Species'] = df['Species'].replace(['Iris-setosa', 'Iris-versicolor', 'Iris-virginica'], [0, 1, 2] )


#q6
print(df.head(10))


#q7
train, test = train_test_split(df, test_size=0.3)
X_train, Y_train = train[["SepalLengthCm", "SepalWidthCm", "PetalLengthCm", "PetalWidthCm"]], train.Species
X_test, Y_test = test[["SepalLengthCm", "SepalWidthCm", "PetalLengthCm", "PetalWidthCm"]], test.Species


#q8
print(X_train.head(10))
print(Y_train.head(10))
print(X_test.head(10))
print(Y_test.head(10))

#method
def classifying(alpha=1e-05, hidden_layer_sizes=(3,3), solver='lbfgs', max_iter=150, epsilon=0.07, learning_rate="constant", learning_rate_init=0.001):
    clf = MLPClassifier(learning_rate=learning_rate, learning_rate_init=learning_rate_init, hidden_layer_sizes=hidden_layer_sizes, alpha=alpha, solver=solver, max_iter=max_iter, epsilon=epsilon)
    t1 = datetime.now()

    mlp = clf.fit(X_train, Y_train)
    t2 = datetime.now()
    delta = t2 - t1
    ms = delta.total_seconds() * 1000
    print(f"Time learning is {ms} milliseconds")

    if hasattr(mlp, 'loss_curve_'):
        plt.plot(mlp.loss_curve_)
        plt.title("training evolution")
        plt.show()
    #prediction
    prediction = mlp.predict(X_test)
    print(prediction)
    #print(X_test)
    print(Y_test.values)
    accuracy = metrics.accuracy_score(prediction, Y_test.values)
    print("the accurancy is:",accuracy)
    pp_matrix_from_data(Y_test.values, prediction)
    return (accuracy, ms)


#9 and 10 and 11
print(classifying())

#q13
print(classifying(solver="sgd",learning_rate_init=0.7))

#q14

params= [
    {
        "solver":"sgd",
        "learning_rate":"constant",
        "learning_rate_init":0.2,
        "max_iter":150,
    },
      {
        "solver":"sgd",
        "learning_rate":"constant",
        "learning_rate_init":0.7,
        "max_iter":300,
    },
  {
        "solver":"sgd",
        "learning_rate":"invscaling",
        "learning_rate_init":0.2,
        "max_iter":300,
    },
      {
        "solver":"sgd",
        "learning_rate":"invscaling",
        "learning_rate_init":0.7,
        "max_iter":150,
    },{
       "solver" :"adam",
       "learning_rate_init":0.01,
      "max_iter":300,

    },
]


for param in params:
  classifying(**param)


#q15
classifying(solver='sgd',alpha=1e-5,hidden_layer_sizes=(3,3), epsilon=0.07 , max_iter = 1500)

#q16

model = keras.Sequential()
model.add(Dense(len(X_train.columns),input_shape=(len(X_train.columns),),activation='relu'))
model.add(Dense(1,activation='softmax'))
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
model.fit(X_train, Y_train)

model = keras.Sequential()
model.add(SimpleRNN(len(X_train.columns),return_sequences=True, return_state=True))
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
