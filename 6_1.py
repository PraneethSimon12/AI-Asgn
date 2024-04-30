'''Create a dataset (.csv file) having following features- experience of the candidate, written score, interview score and salary. Based on three input features, HR decide the salary of the selected candidates. Using this data, KNN model build a for HR department that can help them decide salaries of the candidates. Predict the salaries for the following candidates, by executing the model (for different values of K):
(a) 5 Yrs experience, 8 written test score, 10 interview score
(b) 8 Yrs experience, 7 written test score, 6 interview score'''

import pandas as pd
from sklearn.neighbors import KNeighborsRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error

# Create a dataset
data = {
    'experience': [2, 5, 3, 8, 1, 10, 6, 4, 7, 9],
    'written_score': [7, 9, 6, 8, 5, 10, 7, 6, 9, 8],
    'interview_score': [6, 8, 7, 7, 4, 9, 8, 5, 10, 7],
    'salary': [45000, 65000, 50000, 80000, 35000, 90000, 70000, 55000, 75000, 85000]
}

df = pd.DataFrame(data)

# Save the dataset as a CSV file
df.to_csv('hr_data.csv', index=False)

# Split the data into features and target
X = df[['experience', 'written_score', 'interview_score']]
y = df['salary']

# Split the data into train and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Create and train the KNN model
knn = KNeighborsRegressor()
knn.fit(X_train, y_train)

# Predict salaries for the given candidates
candidate_a = [5, 8, 10]
candidate_b = [8, 7, 6]

# Predict for different values of K
for k in range(1, 11):
    knn = KNeighborsRegressor(n_neighbors=k)
    knn.fit(X_train, y_train)
    
    predicted_a = knn.predict([candidate_a])
    predicted_b = knn.predict([candidate_b])
    
    print(f"For K={k}")
    print(f"Predicted salary for candidate (a): {predicted_a[0]}")
    print(f"Predicted salary for candidate (b): {predicted_b[0]}")
    print()