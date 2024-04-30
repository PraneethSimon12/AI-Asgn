'''Create a dataset (.csv file) having following features- Graduations percentage, experience of the candidate, written score, interview score and selection. Selection feature is binary in nature and contains the status of the candidate. Also store at least 25 records in this dataset.
Using this data, build a Bayesian learning model for HR department that can help them to decide whether the candidate will be selected or not. Take 80% data as training data and remaining a testing data randomly. Using the built model, predict the status for the following unseen data:
(a) 90 %, 5 Yrs experience, 8 written test score, 10 interview score
(b) 75%, 8 Yrs experience, 7 written test score, 6 interview score
Also calculate the possible classification metrics for the above cases and save these values in the .CSV file.'''

import pandas as pd
from sklearn.naive_bayes import GaussianNB
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

# Create a dataset
data = {
    'graduation_percentage': [85, 90, 75, 80, 92, 88, 78, 95, 70, 82, 89, 91, 87, 76, 84, 93, 79, 81, 86, 94, 77, 83, 88, 90, 92],
    'experience': [3, 5, 2, 7, 1, 4, 6, 8, 2, 5, 3, 6, 4, 7, 2, 8, 5, 3, 6, 7, 4, 2, 5, 6, 8],
    'written_score': [7, 8, 6, 9, 5, 7, 8, 10, 6, 7, 8, 9, 7, 6, 8, 9, 7, 8, 7, 10, 6, 8, 7, 9, 8],
    'interview_score': [6, 10, 7, 8, 4, 9, 7, 9, 5, 8, 7, 10, 6, 7, 8, 8, 6, 9, 7, 9, 5, 7, 8, 10, 9],
    'selection': [1, 1, 0, 1, 0, 1, 1, 1, 0, 1, 1, 1, 0, 0, 1, 1, 0, 1, 1, 1, 0, 1, 1, 1, 1]
}

df = pd.DataFrame(data)

# Save the dataset as a CSV file
df.to_csv('hr_data.csv', index=False)

# Split the data into features and target
X = df[['graduation_percentage', 'experience', 'written_score', 'interview_score']]
y = df['selection']

# Split the data into train and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Create and train the Bayesian model
model = GaussianNB()
model.fit(X_train, y_train)

# Predict the selection status for the given candidates
candidate_a = [90, 5, 8, 10]
candidate_b = [75, 8, 7, 6]

predicted_a = model.predict([candidate_a])
predicted_b = model.predict([candidate_b])

print("Predicted selection status for candidate (a):", predicted_a[0])
print("Predicted selection status for candidate (b):", predicted_b[0])

# Calculate classification metrics
y_pred = model.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
precision = precision_score(y_test, y_pred)
recall = recall_score(y_test, y_pred)
f1 = f1_score(y_test, y_pred)

# Save classification metrics in a CSV file
metrics_data = {
    'Metric': ['Accuracy', 'Precision', 'Recall', 'F1-Score'],
    'Value': [accuracy, precision, recall, f1]
}

metrics_df = pd.DataFrame(metrics_data)
metrics_df.to_csv('classification_metrics.csv', index=False)