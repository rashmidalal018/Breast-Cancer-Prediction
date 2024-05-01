
import keras
from keras.models import Sequential
from keras.layers import Dense, Input
from sklearn import datasets
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

model = Sequential()

# Use Input(shape) instead of input_dim
model.add(Input(shape=(30,)))
model.add(Dense(units=32, activation='relu'))
model.add(Dense(units=16, activation='relu'))
model.add(Dense(units=1, activation='sigmoid'))

model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

# for importing out data
cancer = datasets.load_breast_cancer()

x_train, x_test, y_train, y_test = train_test_split(cancer.data, cancer.target, test_size=0.2, random_state=0)

sc = StandardScaler()
x_train = sc.fit_transform(x_train)
x_test = sc.transform(x_test)

# for fitting
print(model.fit(x_train, y_train, epochs=20, batch_size=50, validation_data=(x_test,y_test)))

# for evaluation and prediction 
predictions = model.predict(x_test)

score = model.evaluate(x_test, y_test)

print("Score is : ",score)