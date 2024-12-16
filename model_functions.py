import numpy as np
import pandas as pd
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import cross_val_score
from xgboost import XGBClassifier
from sklearn.model_selection import KFold
from matplotlib import pyplot as plt

best_params = False
feature_importance = True
reweigh = False


def xgboost_model(X_train, X_test, y_train, X_columns, weights_train, best_params=None, feature_importance=False):
    # Set default parameters if none provided
    if best_params:
        params = best_params
    else:
        params = {'max_depth': 6, 'n_estimators': 100, 'learning_rate': 0.1}

    # Initialize and train the XGBoost classifier
    xgb = XGBClassifier(**params, eval_metric='logloss')  # Removed `use_label_encoder`

    # If reweigh is True, use the weights in the model training
    if reweigh:
        xgb.fit(X_train, y_train, sample_weight=weights_train)
    else:
        xgb.fit(X_train, y_train)

    # Make predictions
    y_pred_xgb = xgb.predict(X_test)

    # Perform cross-validation
    try:
        accuracy_xgb = np.mean(cross_val_score(xgb, X_train, y_train, cv=5))
        print(f"Cross-validated Accuracy of XGBoost model: {accuracy_xgb}")
    except Exception as e:
        print(f"Error during cross-validation: {e}")
        accuracy_xgb = None

    # Plot feature importance if requested
    if feature_importance:
        try:
            xgb_importance = pd.Series(xgb.feature_importances_, index=X_columns).sort_values(ascending=False)
            print("Feature importance for XGBoost:")
            print(xgb_importance)
            xgb_importance.plot(kind='bar', color='orange')
            plt.title('Feature Importance for XGBoost')
            plt.ylabel('Importance')
            plt.xlabel('Feature')
            plt.tight_layout()
            plt.show()
        except AttributeError as e:
            print(f"Error displaying feature importance: {e}")

    return y_pred_xgb, accuracy_xgb, xgb


def knn_model(X_train, X_test, y_train, X_columns):

    if best_params:
        params = {}
    else:
        params = {'n_neighbors': 5}

    knn = KNeighborsClassifier(**params)

    # If reweigh is True, use the weights in the model training
    if reweigh:
        knn.fit(X_train, y_train)
    else:
        knn.fit(X_train, y_train)

    y_pred_knn = knn.predict(X_test)
    accuracy_knn = np.mean(cross_val_score(knn, X_train, y_train, cv=5))
    print(f"Cross-validated Accuracy of KNN model: {accuracy_knn}")

    return y_pred_knn, accuracy_knn, knn


def naive_bayes_model(X_train, X_test, y_train, X_columns, weights_train):

    if best_params:
        nb = GaussianNB()
    else:
        nb = GaussianNB()

    # If reweigh is True, use the weights in the model training
    if reweigh:
        nb.fit(X_train, y_train, sample_weight=weights_train)
    else:
        nb.fit(X_train, y_train)

    y_pred_nb = nb.predict(X_test)
    accuracy_nb = np.mean(cross_val_score(nb, X_train, y_train, cv=5))
    print(f"Cross-validated Accuracy of Naive Bayes model: {accuracy_nb}")

    if feature_importance:
        # Feature importance
        nb_importance = pd.Series(nb.theta_[0], index=X_columns).sort_values(ascending=False)
        print("Feature importance for Naive Bayes:")
        print(nb_importance)
        nb_importance.plot(kind='bar', color='lightcoral')
        plt.title('Feature Importance for Naive Bayes')
        plt.ylabel('Importance')
        plt.xlabel('Feature')
        plt.tight_layout()
        plt.show()

    return y_pred_nb, accuracy_nb, nb


def logistic_regression_model(X_train, X_test, y_train, X_columns, weights_train):

    if best_params:
        params = {}
    else:
        params = {'max_iter': 1000}

    log_reg = LogisticRegression(**params)

    # If reweigh is True, use the weights in the model training
    if reweigh:
        log_reg.fit(X_train, y_train, sample_weight=weights_train)
    else:
        log_reg.fit(X_train, y_train)

    y_pred_log_reg = log_reg.predict(X_test)
    accuracy_log_reg = np.mean(cross_val_score(log_reg, X_train, y_train, cv=5))
    print(f"Cross-validated Accuracy of Logistic Regression model: {accuracy_log_reg}")

    if feature_importance:
        # Feature importance
        log_reg_importance = pd.Series(log_reg.coef_[0], index=X_columns).sort_values(ascending=False)
        print("Feature importance for Logistic Regression:")
        print(log_reg_importance)
        log_reg_importance.plot(kind='bar', color='skyblue')
        plt.title('Feature Importance for Logistic Regression')
        plt.ylabel('Importance')
        plt.xlabel('Feature')
        plt.tight_layout()
        plt.show()

    return y_pred_log_reg, accuracy_log_reg, log_reg


def decision_tree_model(X_train, X_test, y_train, X_columns, weights_train):

    if best_params:
        params = {}
    else:
        params = {'max_depth': 5, 'random_state': 42}

    dt = DecisionTreeClassifier(**params)

    # If reweigh is True, use the weights in the model training
    if reweigh:
        dt.fit(X_train, y_train, sample_weight=weights_train)
    else:
        dt.fit(X_train, y_train)

    y_pred_dt = dt.predict(X_test)
    accuracy_dt = np.mean(cross_val_score(dt, X_train, y_train, cv=5))
    print(f"Cross-validated Accuracy of Decision Tree model: {accuracy_dt}")

    if feature_importance:
        # Feature importance
        dt_importance = pd.Series(dt.feature_importances_, index=X_columns).sort_values(ascending=False)
        print("Feature importance for Decision Tree:")
        print(dt_importance)
        dt_importance.plot(kind='bar', color='lightgreen')
        plt.title('Feature Importance for Decision Tree')
        plt.ylabel('Importance')
        plt.xlabel('Feature')
        plt.tight_layout()
        plt.show()

    return y_pred_dt, accuracy_dt, dt


def random_forest_model(X_train, X_test, y_train, X_columns, weights_train):

    if best_params:
        params = {}
    else:
        params = {'max_depth': 5, 'random_state': 42}

    rf = RandomForestClassifier(**params)

    # If reweigh is True, use the weights in the model training
    if reweigh:
        rf.fit(X_train, y_train, sample_weight=weights_train)
    else:
        rf.fit(X_train, y_train)

    y_pred_rf = rf.predict(X_test)
    accuracy_rf = np.mean(cross_val_score(rf, X_train, y_train, cv=5))
    print(f"Cross-validated Accuracy of Random Forest model: {accuracy_rf}")

    if feature_importance:
        # Feature importance
        rf_importance = pd.Series(rf.feature_importances_, index=X_columns).sort_values(ascending=False)
        print("Feature importance for Random Forest:")
        print(rf_importance)
        rf_importance.plot(kind='bar', color='salmon')
        plt.title('Feature Importance for Random Forest')
        plt.ylabel('Importance')
        plt.xlabel('Feature')
        plt.tight_layout()
        plt.show()

    return y_pred_rf, accuracy_rf, rf
