'''For the IRIS dataset, design a decision tree classifier. Take different percentage of training data and then observe effect on the accuracy and other quality parameters. Also note the effect of other decision tree parameters (like max depth, min_sample_spit etc.) on the performance of the model.
Note: Take criterion as entropy.'''

import pandas as pd
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

# Load the IRIS dataset
iris = load_iris()
X = iris.data
y = iris.target

# Convert the data to a pandas DataFrame
data = pd.DataFrame(X, columns=iris.feature_names)
data['target'] = y

# Define the quality metrics function
def calculate_metrics(y_true, y_pred):
    accuracy = accuracy_score(y_true, y_pred)
    precision = precision_score(y_true, y_pred, average='macro')
    recall = recall_score(y_true, y_pred, average='macro')
    f1 = f1_score(y_true, y_pred, average='macro')
    return accuracy, precision, recall, f1

# Evaluate the model for different training data percentages
training_percentages = [0.6, 0.7, 0.8, 0.9]
for percentage in training_percentages:
    X_train, X_test, y_train, y_test = train_test_split(X, y, train_size=percentage, random_state=42)
    
    # Create and train the decision tree classifier
    dt_classifier = DecisionTreeClassifier(criterion='entropy')
    dt_classifier.fit(X_train, y_train)
    
    # Make predictions on the test data
    y_pred = dt_classifier.predict(X_test)
    
    # Calculate the quality metrics
    accuracy, precision, recall, f1 = calculate_metrics(y_test, y_pred)
    
    print(f"Training data percentage: {percentage * 100}%")
    print(f"Accuracy: {accuracy:.2f}")
    print(f"Precision: {precision:.2f}")
    print(f"Recall: {recall:.2f}")
    print(f"F1-Score: {f1:.2f}")
    print()

# Evaluate the model for different decision tree parameters
max_depths = [2, 4, 6, 8, None]
min_samples_splits = [2, 4, 6, 8, 10]

for max_depth in max_depths:
    for min_samples_split in min_samples_splits:
        X_train, X_test, y_train, y_test = train_test_split(X, y, train_size=0.8, random_state=42)
        
        # Create and train the decision tree classifier
        dt_classifier = DecisionTreeClassifier(criterion='entropy', max_depth=max_depth, min_samples_split=min_samples_split)
        dt_classifier.fit(X_train, y_train)
        
        # Make predictions on the test data
        y_pred = dt_classifier.predict(X_test)
        
        # Calculate the quality metrics
        accuracy, precision, recall, f1 = calculate_metrics(y_test, y_pred)
        
        print(f"Max Depth: {max_depth}, Min Samples Split: {min_samples_split}")
        print(f"Accuracy: {accuracy:.2f}")
        print(f"Precision: {precision:.2f}")
        print(f"Recall: {recall:.2f}")
        print(f"F1-Score: {f1:.2f}")
        print()