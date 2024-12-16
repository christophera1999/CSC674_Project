import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.ensemble import VotingClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.model_selection import cross_val_score, train_test_split
from imblearn.over_sampling import SMOTE
import matplotlib.pyplot as plt
import shap
import pickle

# Importing custom functions
from model_functions import knn_model, naive_bayes_model, logistic_regression_model, decision_tree_model, random_forest_model, xgboost_model
from fairness_functions import reweigh_data, evaluate_fairness, calculate_tpr_fpr

# Declare globals
reweigh = False


if __name__ == "__main__":

    ############################# Data Preprocessing #############################

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

    # Include sensitive feature in X for consistent resampling
    X['sensitive_feature'] = sensitive_feature

    ''' Incorporate SMOTE ???? '''

    # Balancing the dataset using SMOTE
    smote = SMOTE(random_state=42)
    #X_resampled, y_resampled = smote.fit_resample(X, y)  # Should be X, y?
    X, y = smote.fit_resample(X, y)

    # Extract sensitive feature after SMOTE
    # sensitive_feature_resampled = X_resampled['sensitive_feature']
    #X_resampled = X_resampled.drop(columns=['sensitive_feature'])

    #sensitive_feature = X['sensitive_feature']
    #X = X.drop(columns=['sensitive_feature'])

    # Apply the Reweighing algorithm to calculate instance weights
    #weights = reweigh_data(X, y, sensitive_feature, y)
    weights = reweigh_data(X, y, X['Gender'], y)

    if reweigh:
        # Performing an 80-20 split using random split
        X_train, X_test, y_train, y_test, weights_train, weights_test = train_test_split(
            X, y, weights, test_size=0.2, random_state=7, stratify=y
        )
    else:
        # Performing an 80-20 split using random split
        weights_train = None
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

    ############################# Model Training and Evaluation #############################

    # Running each model
    y_pred_xgb, accuracy_xgb, xgb_model = xgboost_model(X_train, X_test, y_train, X.columns, weights_train)
    y_pred_knn, accuracy_knn, knn_model = knn_model(X_train, X_test, y_train, X.columns)
    y_pred_nb, accuracy_nb, nb_model = naive_bayes_model(X_train, X_test, y_train, X.columns, weights_train)
    y_pred_log_reg, accuracy_log_reg, log_reg_model = logistic_regression_model(X_train, X_test, y_train, X.columns, weights_train)
    y_pred_dt, accuracy_dt, dt_model = decision_tree_model(X_train, X_test, y_train, X.columns, weights_train)
    y_pred_rf, accuracy_rf, rf_model = random_forest_model(X_train, X_test, y_train, X.columns, weights_train)

    # Ensemble voting classifier using soft voting and cross-validated weights
    voting_clf = VotingClassifier(estimators=[
        #('knn', knn_model),
        ('nb', nb_model),
        ('log_reg', log_reg_model),
        ('dt', dt_model),
        ('rf', rf_model),
        ('xgb', xgb_model)
    ], voting='soft', weights=[accuracy_nb, accuracy_log_reg, accuracy_dt, accuracy_rf, accuracy_xgb]) #*accuracy_knn*#

    # Fit the ensemble voting classifier
    if reweigh:
         voting_clf.fit(X_train, y_train, sample_weight=weights_train)
    else:
        voting_clf.fit(X_train, y_train)

    voting_predictions = voting_clf.predict(X_test)
    voting_accuracy = accuracy_score(y_test, voting_clf.predict(X_test))
    print(f"Accuracy of Ensemble Voting classifier: {voting_accuracy}")
    print(f"\nPredictions of Ensemble Voting classifier:")
    print(voting_predictions)

    #export the model to pkl file
    with open('voting_classifier.pkl', 'wb') as file:
        pickle.dump(voting_clf, file)

    feature_importances = pd.DataFrame(index=X.columns)

    # Extract feature importances from individual classifiers
    for name, model in voting_clf.named_estimators_.items():
        if hasattr(model, "feature_importances_"):  # Tree-based models
            feature_importances[name] = model.feature_importances_
        elif hasattr(model, "coef_"):  # Logistic Regression
            # Take the absolute value of coefficients as importance
            feature_importances[name] = np.abs(model.coef_).flatten()

    # Compute the mean importance across all models
    feature_importances["Mean Importance"] = feature_importances.mean(axis=1)

    # Sort features by mean importance
    feature_importances = feature_importances.sort_values(by="Mean Importance", ascending=False)

    # Plot the feature importances
    plt.figure(figsize=(10, 6))
    feature_importances["Mean Importance"].plot(kind="bar", color="lightgreen")
    plt.title("Feature Importance for Voting Classifier")
    plt.ylabel("Importance")
    plt.xlabel("Feature")
    plt.xticks(rotation=45, ha="right")
    plt.tight_layout()
    plt.show()

    ############################# Fairness analysis by gender #############################

    # Get the Gender column from the original dataset
    X_test_df = pd.DataFrame(X_test, columns=X.columns)
    X_test_df = X_test_df.reset_index(drop=True)

    # Decode Age_Range column
    # X_test_df['Age_Range_Decoded'] = label_encoders['Age_Range'].inverse_transform(X_test_df['Age_Range'])

    # Reset index for the original DataFrame to align with test set
    df_reset = df.reset_index(drop=True)

    # Add Gender column from the original DataFrame to the test set DataFrame
    X_test_df['Gender'] = df_reset.loc[:len(X_test_df) - 1, 'Gender'].values
    # X_test_df['Gender'] = df_reset.loc[X_test_df.index[:len(X_test_df)], 'Gender']

    # Align y_test with X_test_df
    y_test_df = pd.Series(y_test.values, index=X_test_df.index)

    # Align predictions with y_test_df
    y_pred = pd.Series(voting_predictions, index=y_test_df.index)

    # Evaluate fairness metrics
    fairness_metrics, cond_satisfied = evaluate_fairness(
        y_true=y_test_df,
        y_pred=y_pred,
        sensitive_feature=X_test_df['Gender']
    )


    # Evaluate fairness metrics using AIF360
    # fairness_metrics_aif360 = evaluate_fairness_aif360(
    #     y_true=y_test_df,
    #     y_pred=y_pred,
    #     X_test_df=X_test_df,
    #     sensitive_feature_name='Gender'
    # )

    # Graph the fairness metrics
    # Separate male and female groups using the test set before model predictions
    male_indices = X_test_df[X_test_df['Gender'] == 1].index
    female_indices = X_test_df[X_test_df['Gender'] == 0].index

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