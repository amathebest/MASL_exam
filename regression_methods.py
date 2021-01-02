# libraries import
import sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import RepeatedKFold
from sklearn.metrics import mean_squared_error
from sklearn.linear_model import LinearRegression, Ridge, Lasso, ElasticNet
from sklearn.linear_model import RidgeCV, LassoCV

# constants declaration
LAMBDA_VALUES = 10**np.linspace(10, -2, 100)*0.5
MAX_ITER_LASSO = 20000

# this method computes linear regression
def linear_regression(X_train, X_test, y_train, y_test, outputs = False, normalization = False):
    # applying linear regression to the training set
    lin_reg = LinearRegression(normalize = normalization)
    lin_reg.fit(X_train, y_train)

    # predicting y values of the training set
    y_train_predicted = lin_reg.predict(X_train)
    train_set_mse = mean_squared_error(y_train, y_train_predicted)

    # predicting y values of the training set
    y_test_predicted = lin_reg.predict(X_test)
    test_set_mse = mean_squared_error(y_test, y_test_predicted)

    # output and analysis
    if outputs:
        print('Linear regression coefficients:', lin_reg.coef_)
        print('Training test: MSE:', round(train_set_mse, 4), ', R2:', round(lin_reg.score(X_train, y_train), 4))
        print('Test test: MSE:', round(test_set_mse, 4), ', R2:', round(lin_reg.score(X_test, y_test), 4))

    return train_set_mse, test_set_mse

# this method computes ridge regression using the optimal parameter computed with cross validation
def ridge_regression(X_train, X_test, y_train, y_test, output = False, normalization = False):
    # storing the values of the coefficients and the corresponding value of mse for each value of lambda
    ridge = Ridge(normalize = normalization)
    coefficients = []
    mse_training = []
    for lambda_value in LAMBDA_VALUES:
        ridge.set_params(alpha = lambda_value)
        ridge.fit(X_train, y_train)
        # storing the values for each value of lambda for the plots
        coefficients.append(ridge.coef_)
        mse_training.append(mean_squared_error(y_train, ridge.predict(X_train)))

    # definel model evaluation method
    cv = RepeatedKFold(n_splits = 10, n_repeats = 3, random_state = 1)

    # using cross validation to determine best value for lambda
    cv_ridge = RidgeCV(alphas = LAMBDA_VALUES, cv = cv, scoring = 'neg_mean_squared_error', normalize = normalization)
    cv_ridge.fit(X_train, y_train)
    optimal_lambda = cv_ridge.alpha_

    # fitting the ridge regression model with the optimal value of lambda found with cross validation (minimum score value)
    ridge_model = Ridge(alpha = optimal_lambda, normalize = normalization)
    ridge_model.fit(X_train, y_train)

    # predicting y values of the training set
    y_train_predicted = ridge_model.predict(X_train)
    train_set_mse = mean_squared_error(y_train, y_train_predicted)

    # predicting y values of the test set
    y_test_predicted = ridge_model.predict(X_test)
    test_set_mse = mean_squared_error(y_test, y_test_predicted)

    # output and analysis
    if output:
        print('Ridge regression coefficients:', ridge_model.coef_)
        print('Optimal lambda:', optimal_lambda)
        print('Training test: MSE:', round(train_set_mse, 4), ', R2:', round(ridge_model.score(X_train, y_train), 4))
        print('Test test: MSE:', round(test_set_mse, 4), ', R2:', round(ridge_model.score(X_test, y_test), 4))

        # plot that visualizes the coefficients getting shrinked
        ax = plt.gca()
        ax.plot(np.log(LAMBDA_VALUES), coefficients)
        plt.axis('tight')
        plt.xlabel('log(λ)')
        plt.ylabel('Coefficients')
        plt.show()

        # plot for optimal value of lambda obtained with cross validation
        plt.plot(np.log(LAMBDA_VALUES), mse_training)
        plt.vlines(x = np.log(optimal_lambda), ymin = 15, ymax = 90, color = 'red', zorder = 2)
        plt.axis('tight')
        plt.xlabel('log(λ)')
        plt.ylabel('MSE')
        plt.title('Optimal value of λ using cross-validation')
        plt.show()

    return train_set_mse, test_set_mse

# this method computes lasso regression using the optimal parameter computed with cross validation
def lasso_regression(X_train, X_test, y_train, y_test, output = True, normalization = False):
    # storing the values of the coefficients and the corresponding value of mse for each value of lambda
    lasso = Lasso(max_iter = MAX_ITER_LASSO, normalize = normalization)
    coefficients = []
    mse_training = []
    for lambda_value in LAMBDA_VALUES:
        lasso.set_params(alpha = lambda_value)
        lasso.fit(X_train, y_train)
        # storing the values for each value of lambda for the plots
        coefficients.append(lasso.coef_)
        mse_training.append(mean_squared_error(y_train, lasso.predict(X_train)))

    # definel model evaluation method
    cv = RepeatedKFold(n_splits = 5, n_repeats = 3, random_state = 1)

    # using cross validation to determine best value for lambda
    cv_ridge = LassoCV(alphas = LAMBDA_VALUES, max_iter = MAX_ITER_LASSO, cv = cv, normalize = normalization)
    cv_ridge.fit(X_train, y_train)
    optimal_lambda = cv_ridge.alpha_

    # fitting the lasso regression model with the optimal value of lambda found with cross validation
    lasso_model = Lasso(alpha = optimal_lambda).fit(X_train, y_train)

    # predicting y values of the training set
    y_train_predicted = lasso_model.predict(X_train)
    train_set_mse = mean_squared_error(y_train, y_train_predicted)

    # predicting y values of the training set
    y_test_predicted = lasso_model.predict(X_test)
    test_set_mse = mean_squared_error(y_test, y_test_predicted)

    # output and analysis
    if output:
        print('Lasso regression coefficients:', lasso_model.coef_)
        print('Optimal lambda:', optimal_lambda)
        print('Training test: MSE:', round(train_set_mse, 4), ', R2:', round(lasso_model.score(X_train, y_train), 4))
        print('Test test: MSE:', round(test_set_mse, 4), ', R2:', round(lasso_model.score(X_test, y_test), 4))

        # plot that visualizes the coefficients getting shrinked
        ax = plt.gca()
        ax.plot(np.log(LAMBDA_VALUES), coefficients)
        plt.axis('tight')
        plt.xlabel('log(λ)')
        plt.ylabel('Coefficients')
        plt.show()

        # plot for optimal value of lambda obtained with cross validation
        plt.plot(np.log(LAMBDA_VALUES), mse_training)
        plt.vlines(x = np.log(optimal_lambda), ymin = 10, ymax = 100, color = 'red', zorder = 2)
        plt.axis('tight')
        plt.xlabel('log(λ)')
        plt.ylabel('MSE')
        plt.title('Optimal value of λ using cross-validation')
        plt.show()

    return train_set_mse, test_set_mse

def adaptive_lasso_regression(X_train, X_test, y_train, y_test):

    return

# this method computes elastic net regression using the optimal parameter computed with cross validation
def elastic_net_regression(X_train, X_test, y_train, y_test):
    # storing the values of the coefficients and the corresponding value of mse for each value of lambda
    elastic_net = ElasticNet(normalize = True)
    coefficients = []
    mse_training = []
    for lambda_value in LAMBDA_VALUES:
        elastic_net.set_params(alpha = lambda_value)
        elastic_net.fit(X_train, y_train)
        # storing the values for each value of lambda for the plots
        coefficients.append(elastic_net.coef_)
        mse_training.append(mean_squared_error(y_train, lasso.predict(X_train)))


    # fitting the elastic net regression model with the optimal value of lambda found with cross validation
    elastic_net_model = ElasticNet(alpha = 0.005).fit(X_train, y_train)

    # predicting y values of the training set
    y_train_predicted = elastic_net_model.predict(X_train)
    train_set_mse = mean_squared_error(y_train, y_train_predicted)

    # predicting y values of the training set
    y_test_predicted = elastic_net_model.predict(X_test)
    test_set_mse = mean_squared_error(y_test, y_test_predicted)

    # output and analysis
    print('Elastic Net regression coefficients:', elastic_net_model.coef_)
    print('Training test: MSE:', round(train_set_mse, 4), ', R2:', round(elastic_net_model.score(X_train, y_train), 4))
    print('Test test: MSE:', round(test_set_mse, 4), ', R2:', round(elastic_net_model.score(X_test, y_test), 4))

    return train_set_mse, test_set_mse
