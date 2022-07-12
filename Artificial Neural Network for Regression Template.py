from pandas import read_csv
from keras.models import Sequential
from keras.layers import Dense
from keras.wrappers.scikit_learn import KerasRegressor
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import KFold
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline

# load dataset
dataframe = read_csv("housing.data", delim_whitespace=True, header=None)
dataset = dataframe.values

# split into input (X) and output (Y) variables
X = dataset[:,0:13]
Y = dataset[:,13]

#define the model
def larger_model():
 #create model
 model = Sequential()
 model.add(Dense(20, input_dim=13, kernel_initializer='normal', activation='relu'))
 model.add(Dense(6, kernel_initializer='normal', activation='relu'))
 model.add(Dense(1, kernel_initializer='normal'))
 #Compile model
 model.compile(loss='mean_squared_error', optimizer='adam')
 return model

#evaluate model with standardized dataset
estimators = []
estimators.append(('standardize', StandardScaler()))
estimators.append(('mlp', KerasRegressor(build_fn=larger_model, epochs=100, batch_size=15, verbose=1)))

pipeline = Pipeline(estimators)
kfold = KFold(n_splits=10)

results = cross_val_score(pipeline, X, Y, cv=kfold)
print("Larger: %.2f (%.2f) MSE" % (results.mean(), results.std()))
#Since its an MSE the smaller the value the better model is

#removing the negative sign
results_mse = -results
import numpy as np

#converting from MSE to RMSE (Root MSE)
results_rmse = np.sqrt(results_mse)
print(results_rmse)

#calculate the avarage RMSE
results_rmse.mean()
#one-line
results = np.sqrt(-cross_val_score(pipeline, X, Y, cv=kfold)).mean()