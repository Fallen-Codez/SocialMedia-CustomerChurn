import pandas as pd
import matplotlib.pyplot as plt
import warnings
from sklearn.model_selection import train_test_split as ttp
from sklearn.tree import DecisionTreeClassifier as dtc
from sklearn.metrics import accuracy_score
from sklearn import tree

warnings.filterwarnings('ignore')

# Simulating a synthetic dataset for the social media platform, once again I should kaggle this at some point
data = {
    'UserID': range(1, 101),
    'Age': [10, 15, 20, 25, 30, 35, 40, 45, 55, 60] * 10,  # Age of customers
    'Time_Spent': [30, 45, 60, 75, 90, 105, 120, 135, 150, 165] * 10,  # Time spent on the app (in minutes)
    'SupportTickets': [0, 1, 2, 3, 4, 5, 6, 7, 8, 9] * 10,  # Number of support tickets
    'Churn': ['No', 'No', 'Yes', 'No', 'Yes', 'No', 'Yes', 'Yes', 'No', 'Yes'] * 10  # Churn status
}
df = pd.DataFrame(data)

# Splitting the dataset into features [Age, Time_Spent, and SupportTickets] and target variable [Churn]

X = df[['Age', 'Time_Spent', 'SupportTickets']]
y = df['Churn']

# Train and Test split - 7:3 ratio
X_train, X_test, y_train, y_test = ttp(X, y, test_size=0.3, random_state=42)

# Training the Decision Tree model loaded from earlier
clf = dtc()
clf.fit(X_train, y_train)  # access dtc fit method to train

# Making predictions on the test set by using the model (clf) trained above
y_predict = clf.predict(X_test)

# Accuracy is for evaluation of correct predictions

accuracy = accuracy_score(y_test, y_predict)  # y_test: true val of X_test vs y_predict [X_test]
print(f'Model Accuracy: {accuracy}')

# Visualizing the decision tree to plot the model thought process
plt.figure(figsize=(12, 8))
tree.plot_tree(clf, filled=True, feature_names=['Age', 'Time_Spent', 'SupportTickets'],
               class_names=['No Churn', 'Churn'])
plt.title('Decision Tree for Predicting Social Media Customer Churn')
plt.show()

# Mohammed Shuhaib - Last update: 06/09/2024
