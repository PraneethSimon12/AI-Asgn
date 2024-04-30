'''By taking Classified Data as input, compare the performance of KNN, Bayesian Classifier and Decision Tree model. In this comparison, you can take different
possible parameters of particular model. Save these output in a .csv file.'''

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

# Load your classified dataset
# Replace 'your_dataset.csv' with the path to your dataset file
data = pd.read_csv('your_dataset.csv')

# Separate features and target variable
X = data.drop('target', axis=1)
y = data['target']

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Define a function to calculate metrics
def calculate_metrics(y_true, y_pred):
    accuracy = accuracy_score(y_true, y_pred)
    precision = precision_score(y_true, y_pred, average='macro')
    recall = recall_score(y_true, y_pred, average='macro')
    f1 = f1_score(y_true, y_pred, average='macro')
    return accuracy, precision, recall, f1

# KNN model
knn_params = [1, 3, 5, 7, 9]
knn_metrics = []

for k in knn_params:
    knn = KNeighborsClassifier(n_neighbors=k)
    knn.fit(X_train, y_train)
    y_pred = knn.predict(X_test)
    accuracy, precision, recall, f1 = calculate_metrics(y_test, y_pred)
    knn_metrics.append([k, accuracy, precision, recall, f1])

# Bayesian Classifier
bayes = GaussianNB()
bayes.fit(X_train, y_train)
y_pred = bayes.predict(X_test)
accuracy, precision, recall, f1 = calculate_metrics(y_test, y_pred)
bayes_metrics = [['Bayesian', accuracy, precision, recall, f1]]

# Decision Tree
dt_params = [2, 4, 6, 8, None]
dt_metrics = []

for max_depth in dt_params:
    dt = DecisionTreeClassifier(max_depth=max_depth, criterion='entropy')
    dt.fit(X_train, y_train)
    y_pred = dt.predict(X_test)
    accuracy, precision, recall, f1 = calculate_metrics(y_test, y_pred)
    dt_metrics.append([max_depth, accuracy, precision, recall, f1])

# Create a DataFrame with all metrics
metrics_df = pd.DataFrame(knn_metrics, columns=['Parameter', 'Accuracy', 'Precision', 'Recall', 'F1-Score'])
metrics_df = metrics_df.append(pd.DataFrame(bayes_metrics, columns=['Model', 'Accuracy', 'Precision', 'Recall', 'F1-Score']), ignore_index=True)
metrics_df = metrics_df.append(pd.DataFrame(dt_metrics, columns=['Max Depth', 'Accuracy', 'Precision', 'Recall', 'F1-Score']), ignore_index=True)

# Save the metrics to a CSV file
metrics_df.to_csv('model_metrics.csv', index=False)