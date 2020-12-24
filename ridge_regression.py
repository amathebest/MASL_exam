# regression libraries
from sklearn.metrics import mean_squared_error
from sklearn.linear_model import RidgeCV, Ridge

def compute_best_lambda_ridge(X_train, y_train, plot = False):
    # fitting ridge regression with cross validation to obtain the best value of lambda
    alpha_values = [0.0001, 0.001, 0.01, 0.1, 1, 10, 100]
    cv_ridge_model = RidgeCV(alphas = alpha_values, store_cv_values = True).fit(X_train, y_train)
    # if plot is true this also plots the
    optimal_lambda = cv_ridge_model.alpha_
    return optimal_lambda

def compute_ridge_regression(X_train, X_test, y_train, y_test, optimal_lambda):
    # fitting ridge regression model with optimal value of lambda
    ridge_reg = Ridge(alpha = optimal_lambda).fit(X_train, y_train)

    # predicting y values of the training set
    y_train_predicted = ridge_reg.predict(X_train)
    train_set_mse = mean_squared_error(y_train, y_train_predicted)

    # predicting y values of the training set
    y_test_predicted = ridge_reg.predict(X_test)
    test_set_mse = mean_squared_error(y_test, y_test_predicted)

    return train_set_mse, test_set_mse
