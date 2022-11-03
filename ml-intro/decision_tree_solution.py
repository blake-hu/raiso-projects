# Import Pandas library for data science
import pandas as pd
import matplotlib.pyplot as plt

# Read CSV dataset into a Pandas DataFrame
dataset = pd.read_csv("./unbalanced_data.csv")

# Split dataset into attributes (X) and labels (y)
X = dataset.drop('Hire', axis=1)
y = dataset['Hire']

# Randomly split dataset into training set and test set. Test set is 20% of total dataset.
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20)

# Instantiate decision tree classifier and fit it to training dataset
from sklearn.tree import DecisionTreeClassifier
classifier = DecisionTreeClassifier()
classifier.fit(X_train, y_train)

# Use fitted classifier to predict y-values (labels) from test set X-values (attributes)
y_pred = classifier.predict(X_test)
X_test = X_test.set_index(pd.Index(range(X_test.size//2)))
y_pred = pd.DataFrame(y_pred, columns=['Hire'])
X_y_test = X_test.join(y_pred)

# Print raw data for test dataset predictions, separated by Gender and Hire status
hired_m = X_y_test[(X_y_test.Hire==1) & (X_y_test.Gender==1)]
hired_w = X_y_test[(X_y_test.Hire==1) & (X_y_test.Gender==0)]
not_hired_m = X_y_test[(X_y_test.Hire==0) & (X_y_test.Gender==1)]
not_hired_w = X_y_test[(X_y_test.Hire==0) & (X_y_test.Gender==0)]
print('Hired men:\n', hired_m)
print('\nHired women:\n', hired_w)
print('\nNot hired men:\n', not_hired_m)
print('\nNot hired women:\n', not_hired_w, '\n\n')

# Print metrics on accuracy of classifier by comparing predicted and actual y-values of test set
from sklearn.metrics import classification_report, confusion_matrix
print('Confusion matrix:\n', confusion_matrix(y_test, y_pred))
print('\nClassification report:\n', classification_report(y_test, y_pred))