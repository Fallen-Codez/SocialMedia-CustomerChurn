# SocialMedia Customer Churn Model

## Project Overview 
This project involves predicting customer churn for a social media platform using a Decision Tree classifier. The dataset, which is synthetically generated, includes the following features: UserID, Age, Time Spent on Platform, Support Tickets Raised, and a target variable Churn. In this dataset, a churn value of 'Yes' indicates that the customer has stopped using the social media platform.

The dataset is divided into training and testing sets with a 70:30 ratio. The Decision Tree classifier is trained on the training set, and its performance is evaluated on the test set. The model's accuracy is calculated by comparing the predicted values against the true y_test values. The decision-making process of the model is visualised through a decision tree plot using matplotlib.

## Libraries and Dependencies
- pandas
- matplotlib
- warnings
- scikit-learn (for train_test_split, DecisionTreeClassifier, accuracy_score, and tree)

## Decision Tree Visualization
<!--- ![scatter plot](https://github.com/user-attachments/assets/0fd82ff9-b695-40c7-b7f3-1b9b3eb7af93) --->

<div align="center">
<img src="https://github.com/user-attachments/assets/c524ef53-d072-4737-b0bc-60cc1c5056e3" alt="Decision Tree" width="auto" height="auto">
</div>
