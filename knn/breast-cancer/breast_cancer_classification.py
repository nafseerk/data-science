import numpy as np
from sklearn import preprocessing, cross_validation, neighbors
import pandas as pd

# Data clening
df = pd.read_csv('<input_file_name_here>')
df.replace('?', -99999, inplace=True) #The dataset description says there are a lot of ?s in data
df.drop(['id'], 1, inplace=True)

# creating training and test data
X = np.array(df.drop(['class'], 1))
y = np.array(df['class'])
X_train, X_test, y_train, y_test = cross_validation.train_test_split(X, y, test_size=0.2)

# training
clf = neighbors.KNeighborsClassifier()
clf.fit(X_train, y_train)
accuracy = clf.score(X_test, y_test)
#print(accuracy)

# Predicting the classification(benign or malignant) for new cancer patient
example_measures = np.array([4,2,1,1,1,2,3,2,1])
example_measures = example_measures.reshape(1, -1)
prediction = clf.predict(example_measures)
print(prediction)
