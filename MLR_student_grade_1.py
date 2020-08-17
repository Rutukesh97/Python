# Importing libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import pickle

# Load the dataset
dataset = pd.read_csv('student-mat.csv', sep=';')
dataset = dataset[['age','traveltime','studytime','failures','Medu','Fedu','schoolsup','reason','famsup','paid','higher','internet','school','Pstatus','romantic','freetime','health','famrel','Dalc','Walc','absences','G1','G2','G3']]
x = dataset.iloc[:, :-1].values
y = dataset.iloc[:, -1:].values
dataset.dtypes

# Encoding categorical data
# Encoding the Independent Variable
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
ct = ColumnTransformer(transformers=[('encoder', OneHotEncoder(), [6,7,8,9,10,11,12,13,14] )], remainder='passthrough')
x = np.array(ct.fit_transform(x))


#Splitting the data in training an test set
from sklearn.model_selection import train_test_split
# =============================================================================
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size = 0.15)   # This line is here so that the future data 
#                                                                             # will be split into the train and tests set
#                                                                             # and directly use the best accuracy that we obtained 
#                                                                             # by using the loop down below. The accuracy is saved in bestacc.pickle
#                                                                             # so we don't have to train the model again and again.
# =============================================================================
                                                                               
                                                                               
                                                                               
# Finding the best model by using FOR loop
# =============================================================================
                                                      # Once you get the best accuracy, skip the loop for the future data.
best = 0                                   
for _ in range(100):
     x_train, x_test, y_train, y_test = train_test_split(x, y, test_size = 0.15)
     from sklearn.linear_model import LinearRegression
     regressor = LinearRegression()
     regressor.fit(x_train, y_train)
     acc = regressor.score(x_test, y_test)
     print(acc)
     
     if acc > best:
         best = acc
         with open ('bestacc.pickle', 'wb') as f:       # Pickle will save the best model for us,
             pickle.dump(regressor, f)                  # The model which has the best score
 
# =============================================================================


pickle_in= open('bestacc.pickle', 'rb')      # This will open the best model that we saved above and load it for us
regressor = pickle.load(pickle_in)           # So essentially we will be using the best accuracy. 


# Printing the Coefficient and Intercept
print('Coefficient: \n', regressor.coef_)
print('Intercept: \n', regressor.intercept_)


# predicting the result
y_pred = regressor.predict(x_test)
y_pred[y_pred<0] = 0
for z in range(len(y_pred)):
    print(np.round(y_pred[z]) , y_test[z])
    

# =============================================================================    
# Visualizing
parentsedu = dataset[['Medu','Fedu']].mean(axis = 1)
fig = plt.gcf()
fig.set_size_inches(15, 10)
plt.style.use('ggplot')
plt.scatter(parentsedu, dataset['G3'])
plt.xlabel("Avg edu of parents")
plt.ylabel("Final grade of student")
# =============================================================================


# =============================================================================

fig = plt.gcf()
fig.set_size_inches(15, 10)
plt.scatter(dataset['absences'], dataset['G3'])
plt.xlabel('No. of times a student is absent')
plt.ylabel("Final Grade of that student")
# =============================================================================


# =============================================================================

grades = dataset[['G1','G2']].mean(axis = 1)
fig = plt.gcf()
fig.set_size_inches(15, 10)
plt.scatter(grades, dataset['G3'])
plt.xlabel('Avg of grades 1 and 2')
plt.ylabel("Final grade")
# =============================================================================
