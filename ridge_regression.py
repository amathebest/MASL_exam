# libraries import
import sys
import numpy as np
from sklearn.metrics import mean_squared_error
from sklearn.linear_model import RidgeCV, Ridge

# this method computes ridge regression using the optimal parameter computed with cross validation
def compute_ridge_regression(X_train, X_test, y_train, y_test):
    # fitting ridge regression with cross validation to obtain the best value of lambda
    lambda_values = np.arange(0.0001, 1000, 0.1)
    cv_ridge_model = RidgeCV(alphas = lambda_values, scoring = 'neg_mean_squared_error', cv = 5).fit(X_train, y_train)

    # fitting the ridge regression model with the optimal value of lambda found with cross validation
    ridge_model = Ridge(alpha = cv_ridge_model.alpha_).fit(X_train, y_train)

    # predicting y values of the training set
    y_train_predicted = ridge_model.predict(X_train)
    train_set_mse = mean_squared_error(y_train, y_train_predicted)

    # predicting y values of the training set
    y_test_predicted = ridge_model.predict(X_test)
    test_set_mse = mean_squared_error(y_test, y_test_predicted)

    # output and analysis
    print('Ridge regression coefficients:', ridge_model.coef_)
    print('Optimal lambda:', cv_ridge_model.alpha_)
    print('Training test: MSE:', round(train_set_mse, 4), ', R2:', round(ridge_model.score(X_train, y_train), 4))
    print('Test test: MSE:', round(test_set_mse, 4), ', R2:', round(ridge_model.score(X_test, y_test), 4))

    return train_set_mse, test_set_mse
