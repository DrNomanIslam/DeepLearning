from keras.models import *
from keras.layers import *
from imblearn.over_sampling import RandomOverSampler

import numpy as np
seed = 10
np.random.seed(seed)
data = np.genfromtxt('cs-training.csv',delimiter=",",skip_header=1)
data = np.nan_to_num(data)
test_data = data[120000:,:]
data=data[:120000,:]
X=data[:,2:]
Y=data[:,1]
ros = RandomOverSampler()
X,Y=ros.fit_sample(X,Y)
X_test=test_data[:,2:]
Y_test=test_data[:,1]
X_test,Y_test = ros.fit_sample(X_test,Y_test)
features = len(X[0])
m = Sequential()
m.add(Dense(15,input_dim=features,init='uniform',activation='relu'))
m.add(Dense(15,init='uniform',activation='tanh'))
m.add(Dense(1,init='uniform',activation='sigmoid'))
m.compile(loss='binary_crossentropy',optimizer='adam',metrics=['accuracy'])
m.fit(X,Y,nb_epoch=5,batch_size=100)
score = m.evaluate(X_test,Y_test)
print(score)
print("\n\n",m.predict(np.array([[0.766126609,45,2,0.802982129,9120,13,0,6,0,2]])))

# serialize model to JSON
model_json = m.to_json()
with open("model.json", "w") as json_file:
    json_file.write(model_json)
# serialize weights to HDF5
m.save_weights("model.h5")
print("Saved model to disk")
