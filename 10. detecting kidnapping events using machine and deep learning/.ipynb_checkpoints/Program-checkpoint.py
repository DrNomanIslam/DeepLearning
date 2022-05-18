import numpy as np
from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten, Reshape, Conv1D

data = np.loadtxt('data.csv',delimiter=',')
X = np.empty((100,50,3))
Y = np.empty(100)
for i in range(0,100):
    X[i]=data[i*50:(i+1)*50,0:3]
    Y[i]=data[i*50,3]

model = Sequential()
model.add(Conv1D(32, 7, activation='relu',input_shape=(50,3)))
model.add(Flatten())
model.add(Dense(1, activation='sigmoid'))
print(model.summary())

model.compile(loss='binary_crossentropy',
                optimizer='adam', metrics=['accuracy'])

model.fit(X,Y,epochs=10,validation_split=0.2,verbose=1)

model.save('model.h5')

