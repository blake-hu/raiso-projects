# TODO: Import Pandas module and bind it to the name "pd"
# <Your code>

# Read CSV dataset into a Pandas DataFrame
dataset = pd.read_csv("./unbalanced_data.csv")

# Split dataset into attributes (X) and labels (y)
X = dataset.drop('Hire', axis=1)
y = dataset['Hire']

# TODO: Randomly split dataset into training set and test set. Test set is 20% of total dataset.
# Hint: Look up the documentation of train_test_split()
from sklearn.model_selection import train_test_split
# <Your code>

# Instantiate a classifier object from the DecisionTreeClassifier class
from sklearn.tree import DecisionTreeClassifier
classifier = DecisionTreeClassifier()

# TODO: Fit classifier to X and y of training set using the fit() method of the classifier object
# Hint: Look up the documentation of DecisionTreeClassifier
# <Your code>

# Use fitted classifier to predict y-values (labels) from test set X-values (attributes)
y_pred = classifier.predict(X_test)
X_test = X_test.set_index(pd.Index(range(X_test.size//2)))
y_pred = pd.DataFrame(y_pred, columns=['Hire'])
X_y_test = X_test.join(y_pred)
print('X_y_test:\n', X_y_test)

# Print raw data for test dataset predictions, separated by Gender and Hire status
hired_m = X_y_test[(X_y_test.Hire==1) & (X_y_test.Gender==1)]
hired_w = X_y_test[(X_y_test.Hire==1) & (X_y_test.Gender==0)]
print('Hired men:\n', hired_m)
print('\nHired women:\n', hired_w)

# TODO: Extract DataFrames for men and women who were not hired 
# <Your code>

# TODO: Print DataFrames for men and women who were not hired, adding a short description before
# <Your code>

# Print metrics on accuracy of classifier by comparing predicted and actual y-values of test set
from sklearn.metrics import classification_report, confusion_matrix
print('Confusion matrix:\n', confusion_matrix(y_test, y_pred))

# TODO 1: Import the function classification_report() from the metrics interface of the sklearn module, like above
# TODO 2: Call classification_report() on actual and predicted y-values, and print the result with a description
# Hint: Look up the documentation for classification_report()
# <Your code>