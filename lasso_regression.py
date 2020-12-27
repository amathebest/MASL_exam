# libraries import
import sys
import numpy as np
from sklearn.metrics import mean_squared_error
from sklearn.linear_model import LassoCV, Lasso

# this method computes lasso regression using the optimal parameter computed with cross validation
def compute_lasso_regression(X_train, X_test, y_train, y_test):
    # fitting lasso regression with cross validation to obtain the best value of lambda
    lambda_values = np.arange(0.0001, 1000, 0.1)
    #cv_lasso_model = LassoCV(alphas = lambda_values, n_jobs = -1, cv = 5).fit(X_train, y_train)

    # fitting the lasso regression model with the optimal value of lambda found with cross validation
    lasso_model = Lasso(alpha = 0.5).fit(X_train, y_train)

    # predicting y values of the training set
    y_train_predicted = lasso_model.predict(X_train)
    train_set_mse = mean_squared_error(y_train, y_train_predicted)

    # predicting y values of the training set
    y_test_predicted = lasso_model.predict(X_test)
    test_set_mse = mean_squared_error(y_test, y_test_predicted)

    # output and analysis
    print('Lasso regression coefficients:', lasso_model.coef_)
    #print('Optimal lambda:', cv_lasso_model.alpha_)
    print('Training test: MSE:', round(train_set_mse, 4), ', R2:', round(lasso_model.score(X_train, y_train), 4))
    print('Test test: MSE:', round(test_set_mse, 4), ', R2:', round(lasso_model.score(X_test, y_test), 4))

    return train_set_mse, test_set_mse
