import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.model_selection import cross_val_score, train_test_split
from sklearn.linear_model import LogisticRegression
from imblearn.over_sampling import SMOTE

from fairness_functions import evaluate_fairness

# Declare globals for settings
fairness = True


def hyperparameter_tuning_lr(X_train, y_train, X_test, y_test, X_columns):
    best_hyperparameters = {}
    best_fair_hyperparameters = {}
    best_accuracy = 0
    best_fair_accuracy = 0

    # Hyperparameter grids for LogisticRegression
    penalty_options = ['l1', 'l2', 'elasticnet', 'none']
    solver_options = ['lbfgs', 'liblinear', 'saga']
    C_values = [0.01, 0.1, 1, 10, 100]

    for penalty in penalty_options:
        for solver in solver_options:
            for C in C_values:
                try:
                    lr = LogisticRegression(
                        penalty=penalty,
                        solver=solver,
                        C=C,
                        max_iter=1000,
                        random_state=42
                    )
                    accuracy = np.mean(cross_val_score(lr, X_train, y_train, cv=5))
                    lr.fit(X_train, y_train)
                    y_pred = lr.predict(X_test)

                    # Adjusted code before calling fairness evaluation
                    X_test_df = pd.DataFrame(X_test, columns=X_columns).reset_index(drop=True)
                    y_test = pd.Series(y_test).reset_index(drop=True)

                    if fairness:
                        # Evaluate fairness metrics
                        _, cond_satisfied = evaluate_fairness(
                            y_true=y_test,
                            y_pred=y_pred,
                            sensitive_feature=X_test_df['Gender']
                        )

                        if cond_satisfied and accuracy > best_fair_accuracy:
                            best_fair_accuracy = accuracy
                            best_fair_hyperparameters = {
                                'penalty': penalty,
                                'solver': solver,
                                'C': C
                            }

                    if accuracy > best_accuracy:
                        best_accuracy = accuracy
                        best_hyperparameters = {
                            'penalty': penalty,
                            'solver': solver,
                            'C': C
                        }
                except Exception as e:
                    continue

    with open("Hyperparamter_tuning/lr_hyperparameters.txt", "w") as file:
        file.write(f"Best Hyperparameters for Logistic Regression: {best_hyperparameters}\n")
        file.write(f"Best Cross-validated Accuracy: {best_accuracy}\n")
        if fairness:
            file.write(f"Best Accuracy with Fairness Conditions Met: {best_fair_accuracy}\n")
            file.write(f"Best Hyperparameters with Fairness Conditions Met: {best_fair_hyperparameters}\n\n")

    return best_hyperparameters, best_accuracy, best_fair_accuracy


if __name__ == "__main__":

    # Load the dataset
    df = pd.read_csv("synthetic_data.csv")
    # Drop rows with NaN values
    initial_row_count = df.shape[0]
    df.replace("", pd.NA, inplace=True)
    df = df.dropna()
    dropped_row_count = initial_row_count - df.shape[0]
    print(f"Number of rows dropped due to NaN values: {dropped_row_count}")

    # Encoding categorical features using LabelEncoder
    label_encoders = {}
    categorical_columns = ['Dependents', 'Education', 'Loan_Status']

    for column in categorical_columns:
        le = LabelEncoder()
        df[column] = le.fit_transform(df[column])
        label_encoders[column] = le

    # Splitting data into features and target variable
    X = df.drop(columns=['Loan_Status', 'Loan_ID'])
    y = df['Loan_Status']
    sensitive_feature = df['Gender']  # Change as needed

    # Balancing the dataset using SMOTE
    smote = SMOTE(random_state=42)
    X, y = smote.fit_resample(X, y)

    # Performing an 80-20 split using random split
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

    # # Scaling the features
    # scaler = StandardScaler()
    # X_train = scaler.fit_transform(X_train)
    # X_test = scaler.transform(X_test)

    # Hyperparameter tuning
    best_hyperparameters, best_accuracy, best_fair_accuracy = hyperparameter_tuning_lr(X_train, y_train, X_test, y_test, X.columns)

    # Running the best Logistic Regression model with tuned hyperparameters
    lr_best = LogisticRegression(**best_hyperparameters, max_iter=1000, random_state=42)
    lr_best.fit(X_train, y_train)
    y_pred_best = lr_best.predict(X_test)
    final_accuracy = np.mean(cross_val_score(lr_best, X_train, y_train, cv=5))

    with open("Hyperparamter_tuning/lr_hyperparameters.txt", "a") as file:
        file.write(f"Final Model Accuracy with Best Hyperparameters: {final_accuracy}\n")

    print(f"Final Model Accuracy with Best Hyperparameters: {final_accuracy}")
    if fairness:
        print(f"Highest Accuracy with Fairness Conditions Met: {best_fair_accuracy}")
        print(f"Best Hyperparameters with Fairness Conditions Met: {best_hyperparameters}")