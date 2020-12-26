# libraries import
import sys
import numpy as np
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import cross_validate
from sklearn.linear_model import Lasso

#
def compute_lasso_regression(X_train, X_test, y_train, y_test):
    # fitting lasso regression with cross validation to obtain the best value of lambda
    lambda_values = np.arange(0.0001, 100, 0.1)
    optimal_lambda = 0
    best_score = sys.maxsize
    best_model = None

    # using cross validation to pick the best value of lambda for the lasso regression
    for lambda_value in lambda_values:
        # fitting cross validation models to get the optimal value of lambda
        lasso_model_cv = Lasso(alpha = lambda_value)

        cv_results = cross_validate(lasso_model_cv, X_train, y_train, cv = 5)

        if np.mean(cv_results['test_score']) < best_score:
            best_model = lasso_model_cv
            optimal_lambda = lambda_value

    # fitting the lasso regression model with the optimal value of lambda found with cross validation
    lasso_model = Lasso(alpha = 1).fit(X_train, y_train)
    print(lasso_model.coef_)
    # predicting y values of the training set
    y_train_predicted = lasso_model.predict(X_train)
    train_set_mse = mean_squared_error(y_train, y_train_predicted)

    # predicting y values of the training set
    y_test_predicted = lasso_model.predict(X_test)
    test_set_mse = mean_squared_error(y_test, y_test_predicted)

    return train_set_mse, test_set_mse
