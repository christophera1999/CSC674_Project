import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, VotingClassifier
from sklearn.metrics import accuracy_score,precision_score, recall_score, f1_score
from sklearn.model_selection import cross_val_score, train_test_split
from imblearn.over_sampling import SMOTE
import matplotlib.pyplot as plt
import shap

# Load the dataset
df = pd.read_csv("loan_data.csv")

# Drop rows with NaN values
initial_row_count = df.shape[0]
df = df.dropna()
dropped_row_count = initial_row_count - df.shape[0]
print(f"Number of rows dropped due to NaN values: {dropped_row_count}")

# Encoding categorical features using LabelEncoder
label_encoders = {}
categorical_columns = ['Gender', 'Married', 'Dependents', 'Education', 'Self_Employed', 'Property_Area', 'Loan_Status']

for column in categorical_columns:
    le = LabelEncoder()
    df[column] = le.fit_transform(df[column])
    label_encoders[column] = le

# Splitting data into features and target variable
X = df.drop(columns=['Loan_Status', 'Loan_ID'])
y = df['Loan_Status']

# Balancing the dataset using SMOTE
smote = SMOTE(random_state=42)
X, y = smote.fit_resample(X, y)

# Performing an 80-20 split using random split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

# Scaling the features
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)


def knn_model(X_train, X_test, y_train, X_columns):
    knn = KNeighborsClassifier(n_neighbors=5)
    knn.fit(X_train, y_train)
    y_pred_knn = knn.predict(X_test)
    accuracy_knn = np.mean(cross_val_score(knn, X_train, y_train, cv=5))
    print(f"Cross-validated Accuracy of KNN model: {accuracy_knn}")
    return y_pred_knn, accuracy_knn


def naive_bayes_model(X_train, X_test, y_train, X_columns):
    nb = GaussianNB()
    nb.fit(X_train, y_train)
    y_pred_nb = nb.predict(X_test)
    accuracy_nb = np.mean(cross_val_score(nb, X_train, y_train, cv=5))
    print(f"Cross-validated Accuracy of Naive Bayes model: {accuracy_nb}")
    return y_pred_nb, accuracy_nb


def logistic_regression_model(X_train, X_test, y_train, X_columns):
    log_reg = LogisticRegression(max_iter=1000)
    log_reg.fit(X_train, y_train)
    y_pred_log_reg = log_reg.predict(X_test)
    accuracy_log_reg = np.mean(cross_val_score(log_reg, X_train, y_train, cv=5))
    print(f"Cross-validated Accuracy of Logistic Regression model: {accuracy_log_reg}")
    log_reg_importance = pd.Series(log_reg.coef_[0], index=X_columns).sort_values(ascending=False)
    print("Feature importance for Logistic Regression:")
    print(log_reg_importance)
    log_reg_importance.plot(kind='bar', color='skyblue')
    plt.title('Feature Importance for Logistic Regression')
    plt.ylabel('Importance')
    plt.xlabel('Feature')
    plt.tight_layout()
    plt.show()
    return y_pred_log_reg, accuracy_log_reg


def decision_tree_model(X_train, X_test, y_train, X_columns):
    dt = DecisionTreeClassifier(max_depth=5, random_state=42)
    dt.fit(X_train, y_train)
    y_pred_dt = dt.predict(X_test)
    accuracy_dt = np.mean(cross_val_score(dt, X_train, y_train, cv=5))
    print(f"Cross-validated Accuracy of Decision Tree model: {accuracy_dt}")
    dt_importance = pd.Series(dt.feature_importances_, index=X_columns).sort_values(ascending=False)
    print("Feature importance for Decision Tree:")
    print(dt_importance)
    dt_importance.plot(kind='bar', color='lightgreen')
    plt.title('Feature Importance for Decision Tree')
    plt.ylabel('Importance')
    plt.xlabel('Feature')
    plt.tight_layout()
    plt.show()
    return y_pred_dt, accuracy_dt


def random_forest_model(X_train, X_test, y_train, X_columns):
    rf = RandomForestClassifier(random_state=42, max_depth=5)
    rf.fit(X_train, y_train)
    y_pred_rf = rf.predict(X_test)
    accuracy_rf = np.mean(cross_val_score(rf, X_train, y_train, cv=5))
    print(f"Cross-validated Accuracy of Random Forest model: {accuracy_rf}")
    rf_importance = pd.Series(rf.feature_importances_, index=X_columns).sort_values(ascending=False)
    print("Feature importance for Random Forest:")
    print(rf_importance)
    rf_importance.plot(kind='bar', color='salmon')
    plt.title('Feature Importance for Random Forest')
    plt.ylabel('Importance')
    plt.xlabel('Feature')
    plt.tight_layout()
    plt.show()
    return y_pred_rf, accuracy_rf


# Running each model
y_pred_knn, accuracy_knn = knn_model(X_train, X_test, y_train, X.columns)
y_pred_nb, accuracy_nb = naive_bayes_model(X_train, X_test, y_train, X.columns)
y_pred_log_reg, accuracy_log_reg = logistic_regression_model(X_train, X_test, y_train, X.columns)
y_pred_dt, accuracy_dt = decision_tree_model(X_train, X_test, y_train, X.columns)
y_pred_rf, accuracy_rf = random_forest_model(X_train, X_test, y_train, X.columns)

# Comparing predictions between models
comparison_df = pd.DataFrame({
    'KNN': y_pred_knn,
    'Naive Bayes': y_pred_nb,
    'Logistic Regression': y_pred_log_reg,
    'Decision Tree': y_pred_dt,
    'Random Forest': y_pred_rf
}, index=y_test.index)

# Finding rows where predictions differ
differing_predictions = comparison_df[(comparison_df.nunique(axis=1) > 1)]
print("\nRows where predictions differ between models:")
print(differing_predictions)

# Ensemble voting classifier using soft voting and cross-validated weights
voting_clf = VotingClassifier(estimators=[
    ('knn', KNeighborsClassifier(n_neighbors=5)),
    ('nb', GaussianNB()),
    ('log_reg', LogisticRegression(max_iter=1000)),
    ('dt', DecisionTreeClassifier(random_state=42, max_depth=5)),
    ('rf', RandomForestClassifier(random_state=42, max_depth=5))
], voting='soft', weights=[accuracy_knn, accuracy_nb, accuracy_log_reg, accuracy_dt, accuracy_rf])

voting_clf.fit(X_train, y_train)
voting_predictions = voting_clf.predict(X_test)
voting_accuracy = accuracy_score(y_test, voting_clf.predict(X_test))
print(f"Accuracy of Ensemble Voting classifier: {voting_accuracy}")
print(f"\nPredictions of Ensemble Voting classifier:")
print(voting_predictions)

# Plotting feature importance for the ensemble voting classifier
voting_importance = pd.Series(voting_clf.named_estimators_['log_reg'].coef_[0], index=X.columns).sort_values(ascending=False)
print("Feature importance for Ensemble Voting classifier:")
print(voting_importance)
voting_importance.plot(kind='bar', color='orange')
plt.title('Feature Importance for Ensemble Voting Classifier')
plt.ylabel('Importance')
plt.xlabel('Feature')
plt.tight_layout()
plt.show()


#using shap

# Load the dataset
df = pd.read_csv("loan_data.csv")

# Drop rows with NaN values
initial_row_count = df.shape[0]
df = df.dropna()
dropped_row_count = initial_row_count - df.shape[0]
print(f"Number of rows dropped due to NaN values: {dropped_row_count}")

# Encoding categorical features using LabelEncoder
label_encoders = {}
categorical_columns = ['Gender', 'Married', 'Dependents', 'Education', 'Self_Employed', 'Property_Area', 'Loan_Status']

for column in categorical_columns:
    le = LabelEncoder()
    df[column] = le.fit_transform(df[column])
    label_encoders[column] = le

# Splitting data into features and target variable
X = df.drop(columns=['Loan_Status', 'Loan_ID'])
y = df['Loan_Status']

# Balancing the dataset using SMOTE
smote = SMOTE(random_state=42)
X, y = smote.fit_resample(X, y)

# Performing an 80-20 split using random split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

# Scaling the features
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# Running models and using SHAP for feature importance

def run_model_with_shap(model, X_train, X_test, y_train, X_columns, model_name):
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    accuracy = np.mean(cross_val_score(model, X_train, y_train, cv=5))
    print(f"Cross-validated Accuracy of {model_name} model: {accuracy}")

# Running each model
knn_model = KNeighborsClassifier(n_neighbors=5)
naive_bayes_model = GaussianNB()
logistic_regression_model = LogisticRegression(max_iter=1000)
decision_tree_model = DecisionTreeClassifier(max_depth=5, random_state=42)
random_forest_model = RandomForestClassifier(random_state=42, max_depth=5)

run_model_with_shap(knn_model, X_train, X_test, y_train, X.columns, "KNN")
run_model_with_shap(naive_bayes_model, X_train, X_test, y_train, X.columns, "Naive Bayes")
run_model_with_shap(logistic_regression_model, X_train, X_test, y_train, X.columns, "Logistic Regression")
run_model_with_shap(decision_tree_model, X_train, X_test, y_train, X.columns, "Decision Tree")
run_model_with_shap(random_forest_model, X_train, X_test, y_train, X.columns, "Random Forest")

# Ensemble voting classifier using soft voting and cross-validated weights
voting_clf = VotingClassifier(estimators=[
    ('knn', KNeighborsClassifier(n_neighbors=5)),
    ('nb', GaussianNB()),
    ('log_reg', LogisticRegression(max_iter=1000)),
    ('dt', DecisionTreeClassifier(random_state=42, max_depth=5)),
    ('rf', RandomForestClassifier(random_state=42, max_depth=5))
], voting='soft')

voting_clf.fit(X_train, y_train)
voting_predictions = voting_clf.predict(X_test)
voting_accuracy = accuracy_score(y_test, voting_clf.predict(X_test))
print(f"Accuracy of Ensemble Voting classifier: {voting_accuracy}")

# SHAP for Ensemble Voting Classifier (using Logistic Regression as representative)
explainer = shap.LinearExplainer(voting_clf.named_estimators_['log_reg'], X_train)
shap_values = explainer.shap_values(X_train)
shap.summary_plot(shap_values, X_train, feature_names=X.columns, plot_type="bar", show=False)
plt.title('Feature Importance for Ensemble Voting Classifier using SHAP')
plt.tight_layout()
plt.show()


#Male and Female Metrics

# Load the dataset
df = pd.read_csv("loan_data.csv")

# Drop rows with NaN values
initial_row_count = df.shape[0]
df = df.dropna()
dropped_row_count = initial_row_count - df.shape[0]
print(f"Number of rows dropped due to NaN values: {dropped_row_count}")

# Encoding categorical features using LabelEncoder
label_encoders = {}
categorical_columns = ['Gender', 'Married', 'Dependents', 'Education', 'Self_Employed', 'Property_Area', 'Loan_Status']

for column in categorical_columns:
    le = LabelEncoder()
    df[column] = le.fit_transform(df[column])
    label_encoders[column] = le

# Splitting data into features and target variable
X = df.drop(columns=['Loan_Status', 'Loan_ID'])
y = df['Loan_Status']

# Balancing the dataset using SMOTE for target variable only
smote = SMOTE(random_state=42)
X, y = smote.fit_resample(X, y)

# Performing an 80-20 split using random split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

# Scaling the features
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# Train the ensemble voting classifier
voting_clf = VotingClassifier(estimators=[
    ('knn', KNeighborsClassifier(n_neighbors=5)),
    ('nb', GaussianNB()),
    ('log_reg', LogisticRegression(max_iter=1000)),
    ('dt', DecisionTreeClassifier(random_state=42, max_depth=5)),
    ('rf', RandomForestClassifier(random_state=42, max_depth=5))
], voting='soft')

voting_clf.fit(X_train, y_train)

# Evaluate the model on the test set
y_pred = voting_clf.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
print(f"Accuracy of Ensemble Voting classifier: {accuracy}")

# Fairness analysis by gender
# Get the Gender column from the original dataset
X_test_df = pd.DataFrame(X_test, columns=X.columns)
X_test_df = X_test_df.reset_index(drop=True)
df_reset = df.reset_index(drop=True)
X_test_df['Gender'] = df_reset.loc[X_test_df.index[:len(X_test_df)], 'Gender']
y_test_df = pd.Series(y_test, index=X_test_df.index)

# Separate male and female groups using the test set before model predictions
male_indices = X_test_df[X_test_df['Gender'] == 1].index
female_indices = X_test_df[X_test_df['Gender'] == 0].index

# Ensure that indices are within bounds and align predictions correctly
y_test_df = y_test.reset_index(drop=True)
y_pred = pd.Series(y_pred).reset_index(drop=True)

# Extracting predictions for male and female groups using consistent indexing
y_pred_male = y_pred.loc[male_indices]
y_pred_female = y_pred.loc[female_indices]

# Extract true labels for male and female groups
y_test_male = y_test_df.loc[male_indices]
y_test_female = y_test_df.loc[female_indices]

# Calculate fairness metrics for males
accuracy_male = accuracy_score(y_test_male, y_pred_male)
precision_male = precision_score(y_test_male, y_pred_male)
recall_male = recall_score(y_test_male, y_pred_male)
f1_male = f1_score(y_test_male, y_pred_male)

print("\nFairness Metrics for Male Group:")
print(f"Accuracy: {accuracy_male}")
print(f"Precision: {precision_male}")
print(f"Recall: {recall_male}")
print(f"F1 Score: {f1_male}")

# Calculate fairness metrics for females
accuracy_female = accuracy_score(y_test_female, y_pred_female)
precision_female = precision_score(y_test_female, y_pred_female)
recall_female = recall_score(y_test_female, y_pred_female)
f1_female = f1_score(y_test_female, y_pred_female)

print("\nFairness Metrics for Female Group:")
print(f"Accuracy: {accuracy_female}")
print(f"Precision: {precision_female}")
print(f"Recall: {recall_female}")
print(f"F1 Score: {f1_female}")


# Calculate Equalized Odds
# True positive rate (TPR) and False positive rate (FPR) for both groups
def calculate_tpr_fpr(y_true, y_pred):
    tp = np.sum((y_true == 1) & (y_pred == 1))
    fn = np.sum((y_true == 1) & (y_pred == 0))
    fp = np.sum((y_true == 0) & (y_pred == 1))
    tn = np.sum((y_true == 0) & (y_pred == 0))

    tpr = tp / (tp + fn) if (tp + fn) > 0 else 0
    fpr = fp / (fp + tn) if (fp + tn) > 0 else 0
    return tpr, fpr

# Calculate TPR and FPR for male and female groups
tpr_male, fpr_male = calculate_tpr_fpr(y_test_male, y_pred_male)
tpr_female, fpr_female = calculate_tpr_fpr(y_test_female, y_pred_female)

print("\nEqualized Odds Metrics:")
print(f"Male TPR: {tpr_male}, Male FPR: {fpr_male}")
print(f"Female TPR: {tpr_female}, Female FPR: {fpr_female}")

# Plotting fairness metrics
metrics = ['Accuracy', 'Precision', 'Recall', 'F1 Score']
male_metrics = [accuracy_male, precision_male, recall_male, f1_male]
female_metrics = [accuracy_female, precision_female, recall_female, f1_female]

x = np.arange(len(metrics))  # the label locations
width = 0.35  # the width of the bars

fig, ax = plt.subplots()
rects1 = ax.bar(x - width / 2, male_metrics, width, label='Male', color='lightblue')
rects2 = ax.bar(x + width / 2, female_metrics, width, label='Female', color='lightpink')

# Add some text for labels, title and custom x-axis tick labels, etc.
ax.set_ylabel('Scores')
ax.set_title('Fairness Metrics by Gender')
ax.set_xticks(x)
ax.set_xticklabels(metrics)
ax.legend()

fig.tight_layout()
plt.show()