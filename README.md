
# Tree Ensembles and Random Forests - Lab

In this lab, we used scikit-learn to build and compare different tree-based machine learning models for predicting whether a person earns more than $50K based on demographic features.

Dataset Preparation
Loaded the salaries_final.csv dataset
Separated the target variable (Target) from features using pop()
Identified data types of each column
Converted categorical variables into numeric format using pd.get_dummies()
Split the dataset into training and test sets (75/25 split)

1. Decision Tree (Baseline Model)

A single decision tree classifier was trained with:

criterion = gini
max_depth = 5

Key outcomes:
Provided a baseline for comparison
Showed feature importance using .feature_importances_
Visualized important predictors such as:
Age
Education
Occupation

Evaluation:
Predictions were made on test data
Performance was measured using:
Confusion matrix
Classification report
Accuracy score

2. Bagging Classifier (Ensemble Method)

A Bagging model was created using:
Decision Tree as the base estimator
n_estimators = 20

Key idea:
Multiple trees are trained on different random samples of data
Final prediction is based on majority voting

Results:
Improved stability compared to a single decision tree
Evaluated using .score() for both training and test accuracy

3. Random Forest Classifier

A Random Forest model was trained with:
n_estimators = 100
max_depth = 5

Key idea:
Combines bagging with random feature selection
Each tree sees a different subset of features

Results:
Better generalization than a single decision tree
Higher and more stable accuracy
Feature importance revealed strong predictors across the dataset

4. Understanding Individual Trees in a Forest

A smaller random forest was created:
n_estimators = 5
max_features = 10
max_depth = 2

Purpose:
Make individual trees easier to inspect
Increase diversity between trees

Analysis:
Accessed individual trees using:forest_2.estimators_
Compared feature importance between trees

Observation:
Different trees focus on different features
Confirms randomness improves diversity and model robustness
Key Takeaways
A single decision tree is easy to interpret but can overfit
Bagging reduces variance by averaging multiple trees
Random Forest improves performance further by adding feature randomness
Feature importance helps interpret model decisions
Individual trees in a forest can behave very differently

Final Conclusion

This lab demonstrated how ensemble methods significantly improve upon a single decision tree. By combining multiple trees (bagging and random forests), models become more stable, accurate, and robust while still allowing interpretability through feature importance analysis and tree inspection.

