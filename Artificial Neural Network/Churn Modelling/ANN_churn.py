import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

churn = pd.read_csv("C:\\Users\\DHRUBAJIT\Desktop\\Udemy Courses\\Data Science Course\Deep Learning\\1 Artificial Neural Network\\Artificial_Neural_Networks\\Churn_modelling.csv")

#drop unwanted features
churn = churn.drop(['RowNumber','CustomerId','Surname'], axis=1)

#set the predictors and response variable
X = churn.drop(['Exited'], axis=1)
y = churn.Exited


from sklearn.preprocessing import LabelEncoder, OneHotEncoder
le1 = LabelEncoder()
le2 = LabelEncoder()
onehot = OneHotEncoder(categorical_features=[1])
X.Geography = le1.fit_transform(X.Geography)
X.Gender = le2.fit_transform(X.Gender)
X = onehot.fit_transform(X).toarray()

#avoiding dummy variable trap.
X = X[:,1:]


from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
X = sc.fit_transform(X)


import keras
from keras.wrappers.scikit_learn import KerasClassifier
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import Dropout


# plots
# train and test accuracy plot
def accuracy_plot(model, title):
    train_accu = pd.Series(model.history['acc'])
    print("Mean training accuracy: %.2f"%(train_accu.mean()))
    test_accu = pd.Series(model.history['val_acc'])
    print("Mean testing accuracy: %.2f"%(test_accu.mean()))
    
    plt.plot(model.history['acc'])
    plt.plot(model.history['val_acc'])
    plt.title('%s:model accuracy'%(title))
    plt.ylabel('accuracy')
    plt.xlabel('epoch')
    plt.legend(['train', 'test'], loc='lower right')
    plt.savefig("%s.png" %(title))
    plt.show()

# train and test loss plot
def loss_plot(model, title):
    train_accu = pd.Series(model.history['loss'])
    print("Mean training loss: %.2f"%(train_accu.mean()))
    test_accu = pd.Series(model.history['val_loss'])
    print("Mean testing loss: %.2f"%(test_accu.mean()))
    
    plt.plot(model.history['loss'])
    plt.plot(model.history['val_loss'])
    plt.title('%s: model loss' %(title))
    plt.ylabel('loss')
    plt.xlabel('epoch')
    plt.legend(['train', 'test'], loc='upper left')
    plt.show()


################################################################################################################
# 1. BASELINE model
classifier = Sequential()
classifier.add(Dense(output_dim = 6, init='uniform', activation='relu', input_dim=11))
classifier.add(Dense(output_dim=1, init='uniform', activation='sigmoid'))
classifier.compile(optimizer='adam',loss='binary_crossentropy', metrics=['accuracy'])

baseline = classifier.fit(X, y, validation_split=0.25, epochs=150, batch_size=10, verbose=0)

accuracy_plot(baseline, 'baseline model')

loss_plot(baseline,'baseline model')
################################################################################################################


# 2. Kerasclassifier
def build_classifier():
    classifier1 = Sequential()
    classifier1.add(Dense(output_dim = 6, init = 'uniform', activation = 'relu', input_dim = 11))
    classifier1.add(Dense(output_dim = 1, init = 'uniform', activation = 'sigmoid'))
    classifier1.compile(optimizer = 'adam', loss = 'binary_crossentropy', metrics = ['accuracy'])
    return classifier1

classifier1 = KerasClassifier(build_fn = build_classifier)
keras_class = classifier1.fit(X, y, validation_split=0.25, epochs=150, batch_size=10, verbose=0)
    
accuracy_plot(keras_class, 'KerasClassifier model')
loss_plot(keras_class,'KerasClassifier model')
################################################################################################################



# 3. Smaller Model
def create_smaller():
	# create model
	model = Sequential()
	model.add(Dense(3, input_dim=11, kernel_initializer='normal', activation='relu'))
	model.add(Dense(1, kernel_initializer='normal', activation='sigmoid'))
	# Compile model
	model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
	return model

estimators_small = KerasClassifier(build_fn=create_smaller)
keras_small = estimators_small.fit(X, y, validation_split=0.25, epochs=150, batch_size=10, verbose=0)

accuracy_plot(keras_small, 'KerasClassifier Small model')
loss_plot(keras_small,'KerasClassifier Small model')
################################################################################################################


# 4. wide model
def create_wide():
	# create model
	model = Sequential()
	model.add(Dense(12, input_dim=11, kernel_initializer='normal', activation='relu'))
	model.add(Dense(1, kernel_initializer='normal', activation='sigmoid'))
	# Compile model
	model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
	return model

estimators_wide = KerasClassifier(build_fn=create_wide)
keras_wide = estimators_wide.fit(X, y, validation_split=0.25, epochs=150, batch_size=10, verbose=0)

accuracy_plot(keras_wide, 'KerasClassifier Wide model')
loss_plot(keras_wide,'KerasClassifier Wide model')
################################################################################################################


# 5. larger model
def create_larger():
	# create model
	model = Sequential()
	model.add(Dense(6, input_dim=11, kernel_initializer='normal', activation='relu'))
	model.add(Dense(3, kernel_initializer='normal', activation='relu'))
	model.add(Dense(1, kernel_initializer='normal', activation='sigmoid'))
	# Compile model
	model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
	return model

estimators_large = KerasClassifier(build_fn=create_larger)
keras_large = estimators_large.fit(X, y, validation_split=0.25, epochs=150, batch_size=10, verbose=0)

accuracy_plot(keras_large, 'KerasClassifier Large model')
loss_plot(keras_large,'KerasClassifier Large model')
################################################################################################################


# 6. deep model
def create_deep():
	# create model
	model = Sequential()
	model.add(Dense(6, input_dim=11, kernel_initializer='normal', activation='relu'))
	model.add(Dense(3, kernel_initializer='normal', activation='relu'))
	model.add(Dense(2, kernel_initializer='normal', activation='relu'))
	model.add(Dense(1, kernel_initializer='normal', activation='sigmoid'))
	# Compile model
	model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
	return model

estimators_deep = KerasClassifier(build_fn=create_deep)
keras_deep = estimators_deep.fit(X, y, validation_split=0.25, epochs=150, batch_size=10, verbose=0)

accuracy_plot(keras_deep, 'KerasClassifier Deep model')
loss_plot(keras_deep,'KerasClassifier Deep model')
################################################################################################################


# 7. wide and deep model
def create_wide_deep():
	# create model
	model = Sequential()
	model.add(Dense(12, input_dim=11, kernel_initializer='normal', activation='relu'))
	model.add(Dense(6, kernel_initializer='normal', activation='relu'))
	model.add(Dense(3, kernel_initializer='normal', activation='relu'))
	model.add(Dense(1, kernel_initializer='normal', activation='sigmoid'))
	# Compile model
	model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
	return model

estimators_wide_deep = KerasClassifier(build_fn=create_wide_deep)
keras_wide_deep = estimators_wide_deep.fit(X, y, validation_split=0.25, epochs=150, batch_size=10, verbose=0)

accuracy_plot(keras_wide_deep, 'KerasClassifier Wide & Deep model')
loss_plot(keras_wide_deep,'KerasClassifier Wide & Deep model')
################################################################################################################

# Selcted models ---> 2nd and 4th

from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, classification_report,accuracy_score
xtrain,xtest,ytrain,ytest = train_test_split(X,y, random_state=0, test_size=0.25)

# 2nd model
estimators_keras = KerasClassifier(build_fn=build_classifier, epochs=150, batch_size=10, verbose=0)
estimators_keras.fit(xtrain,ytrain)
keras_pred = estimators_keras.predict(xtest)
print("KerasClassifier [2nd Model]")
print("")
print("Confusion matrix......")
print(confusion_matrix(ytest, keras_pred))
print("")
print("Classification Report.....")
print(classification_report(ytest, keras_pred))
################################################################################################################


# 4th model
print("Wide KerasClassifier [4th Model]")
estimators_wide = KerasClassifier(build_fn=create_wide, epochs=150, batch_size=10, verbose=0)
estimators_wide.fit(xtrain,ytrain)
wide_pred = estimators_wide.predict(xtest)
print("Wide KerasClassifier [4th Model]")
print("")
print("Confusion matrix......")
print(confusion_matrix(ytest, wide_pred))
print("")
print("Classification Report.....")
print(classification_report(ytest, wide_pred))
################################################################################################################

# summarize results - model tuning
def print_summary(grid_model):
    print("Best: %f using %s" % (grid_model.best_score_, grid_model.best_params_))
    means = grid_model.cv_results_['mean_test_score']
    stds = grid_model.cv_results_['std_test_score']
    params = grid_model.cv_results_['params']
    for mean, stdev, param in zip(means, stds, params):
        print("%f (%f) with: %r" % (mean, stdev, param))
        
        
        
#Tuning the 4th model, i.e the wide model
#1. Batch_size and epochs
from sklearn.model_selection import GridSearchCV
batch_size = [10, 20, 40, 60, 80, 100]
epochs = [50, 100, 150]
param_grid = dict(batch_size=batch_size, epochs=epochs)
grid = GridSearchCV(estimator=estimators_wide, param_grid=param_grid)
grid_result = grid.fit(xtrain,ytrain)




#2. Optimizer
def final_mod(optimizer='adam'):
	# create model
	model = Sequential()
	model.add(Dense(12, input_dim=11, kernel_initializer='normal', activation='relu'))
	model.add(Dense(1, kernel_initializer='normal', activation='sigmoid'))
	# Compile model
	model.compile(loss='binary_crossentropy', optimizer=optimizer, metrics=['accuracy'])
	return model

optimizer = ['SGD','Adam','RMSprop','Adagrad', 'Adadelta', 'Adamax', 'Nadam']
parameters2 = dict(optimizer=optimizer)
classifier2 = KerasClassifier(build_fn=final_mod, epochs=50 ,batch_size=60 )
grid2 = GridSearchCV(estimator = classifier2, param_grid=parameters2)
grid2 = grid2.fit(xtrain,ytrain)
print_summary(grid2)



#3. Learning_rate and momentum
from keras.optimizers import Nadam
def final_mod3(learn_rate=0.01):
    classifier3 = Sequential()
    classifier3.add(Dense(12, input_dim=11, kernel_initializer='normal', activation='relu'))
    classifier3.add(Dense(1, kernel_initializer='normal', activation='sigmoid'))
    optimizer = Nadam(lr=learn_rate)
    classifier3.compile(optimizer = optimizer, loss = 'binary_crossentropy', metrics = ['accuracy'])
    return classifier3

learn_rate = [0.001, 0.01, 0.1, 0.2, 0.3]
parameters3 = dict(learn_rate=learn_rate)
classifier3 = KerasClassifier(build_fn=final_mod3, epochs=50 ,batch_size=60 )
grid3 = GridSearchCV(estimator = classifier3, param_grid=parameters3, scoring='accuracy')
grid3 = grid3.fit(xtrain,ytrain)
print_summary(grid3)




#4. Network weight initialization
from keras.optimizers import SGD
def final_mod4(init='uniform'):
    classifier4 = Sequential()
    classifier4.add(Dense(12, input_dim=11, kernel_initializer=init, activation='relu'))
    classifier4.add(Dense(1, kernel_initializer=init, activation='sigmoid'))
    optimizer = Nadam(lr=0.01)
    classifier4.compile(optimizer = optimizer, loss = 'binary_crossentropy', metrics = ['accuracy'])
    return classifier4

init_mode = ['uniform', 'lecun_uniform', 'normal', 'zero', 'glorot_normal', 'glorot_uniform', 'he_normal', 'he_uniform']
parameters4 = dict(init=init_mode)

classifier4 = KerasClassifier(build_fn=final_mod4, epochs=50 ,batch_size=60 )
grid4 = GridSearchCV(estimator = classifier4, param_grid=parameters4, scoring='accuracy')
grid4 = grid4.fit(xtrain,ytrain)
print_summary(grid4)



#5. Neuron Activation Function
def final_mod5(activation='relu'):
    classifier5 = Sequential()
    classifier5.add(Dense(output_dim = 12, init = 'glorot_normal', activation = activation, input_dim = 11))
    classifier5.add(Dense(output_dim = 1, init = 'glorot_normal', activation = 'sigmoid'))
    optimizer = Nadam(lr=0.01)
    classifier5.compile(optimizer = optimizer, loss = 'binary_crossentropy', metrics = ['accuracy'])
    return classifier5

activation = ['softmax', 'softplus', 'softsign', 'relu', 'tanh', 'sigmoid', 'hard_sigmoid', 'linear']
parameters5 = dict(activation=activation)
classifier5 = KerasClassifier(build_fn=final_mod5, epochs=50 ,batch_size=60 )
grid5 = GridSearchCV(estimator = classifier5, param_grid=parameters5, scoring='accuracy')
grid5 = grid5.fit(xtrain,ytrain)
print_summary(grid5)



#6. Dropout regularization
def final_mod6(dropout_rate=0.0, weight_constraint=0):
    classifier6 = Sequential()
    classifier6.add(Dense(output_dim = 12, init = 'glorot_normal', activation = 'relu', input_dim = 11,kernel_constraint=maxnorm(weight_constraint)))
    classifier6.add(Dropout(dropout_rate))    
    classifier6.add(Dense(output_dim = 1, init = 'glorot_normal', activation = 'sigmoid'))
    optimizer = Nadam(lr=0.01)
    classifier6.compile(optimizer = optimizer, loss = 'binary_crossentropy', metrics = ['accuracy'])
    return classifier6

from keras.constraints import maxnorm
weight_constraint = [1, 2, 3, 4, 5]
dropout_rate = [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6]
parameters6 = dict(dropout_rate=dropout_rate, weight_constraint=weight_constraint)
classifier6 = KerasClassifier(build_fn=final_mod6, epochs=50 ,batch_size=60 )
grid6 = GridSearchCV(estimator = classifier6, param_grid=parameters6, scoring='accuracy')
grid6 = grid6.fit(xtrain,ytrain)
print_summary(grid6)



#7. No. of neurons
def final_mod7(neurons=1):
    classifier7 = Sequential()
    classifier7.add(Dense(neurons, init = 'glorot_normal', activation = 'relu', input_dim = 11,kernel_constraint=maxnorm(4)))
    classifier7.add(Dropout(0.6))     
    classifier7.add(Dense(output_dim = 1, init = 'glorot_normal', activation = 'sigmoid'))
    optimizer = Nadam(lr=0.01)
    classifier7.compile(optimizer = optimizer, loss = 'binary_crossentropy', metrics = ['accuracy'])
    return classifier7

neurons = [1, 5, 10, 15, 20, 25, 30, 40]
parameters7 = dict(neurons=neurons)
classifier7 = KerasClassifier(build_fn=final_mod7, epochs=50 ,batch_size=60 )
grid7 = GridSearchCV(estimator = classifier7, param_grid=parameters7, scoring='accuracy')
grid7 = grid7.fit(xtrain,ytrain)
print_summary(grid7)


#################################################################################################################
#################################################################################################################

# FINAL MODEL
def final_model1():
    final = Sequential()
    final.add(Dense(40, init = 'glorot_normal', activation = 'relu', input_dim = 11,kernel_constraint=maxnorm(4)))
    final.add(Dropout(0.6))     
    final.add(Dense(output_dim = 1, init = 'glorot_normal', activation = 'sigmoid'))
    optimizer = Nadam(lr=0.01)
    final.compile(optimizer = optimizer, loss = 'binary_crossentropy', metrics = ['accuracy'])
    return final

final_mod = KerasClassifier(build_fn = final_model1)
final_model = final_mod.fit(xtrain, ytrain, validation_split=0.25, epochs=50, batch_size=60, verbose=0)

accuracy_plot(final_model, 'Final model')
loss_plot(final_model,'Final model')


a = KerasClassifier(build_fn=final_model1, epochs=50, batch_size=60, verbose=0)
a.fit(xtrain,ytrain)

final_pred = a.predict(xtest)
print("Final KerasClassifier.....")
print("Training Score: %.3f"%(a.score(xtrain, ytrain)*100))
print("Testing  Score: %.3f"%(accuracy_score(ytest, final_pred)*100))
print("")
print("Confusion matrix......")
print(confusion_matrix(ytest, final_pred))
print("")
print("Classification Report.....")
print(classification_report(ytest, final_pred))

#################################################################################################################
#################################################################################################################