# libraries import
import sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import statsmodels.api as sm
from sklearn.model_selection import RepeatedKFold
from sklearn.metrics import mean_squared_error
from sklearn.linear_model import LinearRegression, Ridge, Lasso, ElasticNet
from sklearn.linear_model import RidgeCV, LassoCV, ElasticNetCV

# constants declaration
LAMBDA_VALUES = 10**np.linspace(10, -3, 100)*0.5
MAX_ITER = 20000

# this method computes linear regression
def linear_regression(X_train, X_test, y_train, y_test, outputs = False):
    # applying linear regression to the training set
    lin_reg = LinearRegression().fit(X_train, y_train)

    # predicting y values of the training set
    y_train_predicted = lin_reg.predict(X_train)
    train_set_mse = mean_squared_error(y_train, y_train_predicted)

    # predicting y values of the training set
    y_test_predicted = lin_reg.predict(X_test)
    test_set_mse = mean_squared_error(y_test, y_test_predicted)

    # output and analysis
    if outputs:
        print('Linear regression coefficients:', np.round(lin_reg.coef_, 4))
        print('Training test: MSE:', round(train_set_mse, 4), ', R2:', round(lin_reg.score(X_train, y_train), 4))
        print('Test test: MSE:', round(test_set_mse, 4), ', R2:', round(lin_reg.score(X_test, y_test), 4))

    return train_set_mse, test_set_mse

# this method computes ridge regression using the optimal parameter computed with cross validation
def ridge_regression(X_train, X_test, y_train, y_test, output = False):
    # storing the values of the coefficients and the corresponding value of mse for each value of lambda
    ridge = Ridge()
    coefficients = []
    mse_training = []
    for lambda_value in LAMBDA_VALUES:
        ridge.set_params(alpha = lambda_value)
        ridge.fit(X_train, y_train)
        # storing the values for each value of lambda for the plots
        coefficients.append(ridge.coef_)
        mse_training.append(mean_squared_error(y_train, ridge.predict(X_train)))

    # definel model evaluation method
    cv = RepeatedKFold(n_splits = 5, n_repeats = 3, random_state = 1)

    # using cross validation to determine best value for lambda
    cv_ridge = RidgeCV(alphas = LAMBDA_VALUES, cv = cv, scoring = 'neg_mean_squared_error')
    cv_ridge.fit(X_train, y_train)
    optimal_lambda = cv_ridge.alpha_

    # fitting the ridge regression model with the optimal value of lambda found with cross validation (minimum score value)
    ridge_model = Ridge(alpha = optimal_lambda)
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
        print('Optimal lambda:', round(optimal_lambda, 4))
        print('Training test: MSE:', round(train_set_mse, 4), ', R2:', round(ridge_model.score(X_train, y_train), 4))
        print('Test test: MSE:', round(test_set_mse, 4), ', R2:', round(ridge_model.score(X_test, y_test), 4))

        # plot that visualizes the coefficients getting shrinked
        ax = plt.gca()
        ax.plot(np.log(LAMBDA_VALUES), coefficients)
        plt.axis('tight')
        plt.xlabel('log(λ)')
        plt.ylabel('Coefficients')
        plt.title('Parameters shrinkage')
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
def lasso_regression(X_train, X_test, y_train, y_test, output = False):
    # storing the values of the coefficients and the corresponding value of mse for each value of lambda
    lasso = Lasso(max_iter = MAX_ITER)
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
    cv_lasso = LassoCV(alphas = LAMBDA_VALUES, cv = cv)
    cv_lasso.fit(X_train, y_train)
    optimal_lambda = cv_lasso.alpha_

    # fitting the lasso regression model with the optimal value of lambda found with cross validation
    lasso_model = Lasso(alpha = optimal_lambda).fit(X_train, y_train)

    # predicting y values of the training set
    y_train_predicted = lasso_model.predict(X_train)
    train_set_mse = mean_squared_error(y_train, y_train_predicted)
    train_set_score = lasso_model.score(X_train, y_train)

    # predicting y values of the training set
    y_test_predicted = lasso_model.predict(X_test)
    test_set_mse = mean_squared_error(y_test, y_test_predicted)
    test_set_score = lasso_model.score(X_test, y_test)

    # output and analysis
    if output:
        print('Lasso regression coefficients:', lasso_model.coef_)
        print('Optimal lambda:', round(optimal_lambda, 4))
        print('Training test: MSE:', round(train_set_mse, 4), ', R2:', round(train_set_score), 4)
        print('Test test: MSE:', round(test_set_mse, 4), ', R2:', round(test_set_score, 4))

        # plot that visualizes the coefficients getting shrinked
        ax = plt.gca()
        ax.plot(np.log(LAMBDA_VALUES), coefficients)
        plt.axis('tight')
        plt.xlabel('log(λ)')
        plt.ylabel('Coefficients')
        plt.title('Parameters shrinkage')
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

# this method computes adaptive lasso regression with weights estimated using OLS
def adaptive_lasso_regression(X_train, X_test, y_train, y_test, output = False):
    # using a different pool of lambda values for adaptive lasso
    adalasso_lambdas = [0.005, 0.01, 0.05, 0.1, 0.15, 0.2, 0.25, 0.3, 0.5, 1]
    training_mse = []
    test_mse = []

    # looping on the values of lambda
    for lambda_value in adalasso_lambdas:
        n_samples, n_features = X_train.shape
        objective_function = lambda beta: 1. / (2 * n_samples) * np.linalg.norm(y_train - np.dot(X_train, beta), ord = 2)**2 + lambda_value * np.sum(np.abs(beta))
        # creating a skeleton for the weights array
        weights = np.ones(n_features)

        # fitting OLS to estimate the weights
        X = sm.add_constant(X_train)
        estimates = sm.OLS(y_train, X).fit()
        # we exlude the intercept since we don't want to add a penalty to it
        idx = 0
        while idx < len(estimates.pvalues)-1:
            if estimates.params[idx+1] < 0.1 and estimates.params[idx+1] > -0.1:
                weights[idx] = sys.maxsize
            else:
                weights[idx] = 1 / np.abs(estimates.params[idx+1])**np.finfo(float).eps
            idx += 1

        # applying weights to the covariates
        weighted_X_train = X_train / weights[np.newaxis, :]
        weighted_X_test = X_test / weights[np.newaxis, :]

        # storing the values of the coefficients and the corresponding value of mse for each value of lambda
        lasso = Lasso(max_iter = MAX_ITER)
        models = []
        coefficients = []
        mse_training = []
        obj_function_values = []
        for lambda_val in LAMBDA_VALUES:
            lasso.set_params(alpha = lambda_val)
            lasso.fit(weighted_X_train, y_train)
            # storing the values for each value of lambda for the plots
            models.append(lasso)
            coefficients.append(lasso.coef_ / weights)
            mse_training.append(mean_squared_error(y_train, lasso.predict(weighted_X_train)))
            obj_function_values.append(objective_function(lasso.coef_))

        adalasso_model = Lasso(alpha = lambda_value)
        adalasso_model.fit(weighted_X_train, y_train)

        # predicting y values of the training set
        y_train_predicted = adalasso_model.predict(weighted_X_train)
        train_set_mse = mean_squared_error(y_train, y_train_predicted)
        train_set_score = adalasso_model.score(weighted_X_train, y_train)

        # predicting y values of the training set
        y_test_predicted = adalasso_model.predict(weighted_X_test)
        test_set_mse = mean_squared_error(y_test, y_test_predicted)
        test_set_score = adalasso_model.score(weighted_X_test, y_test)

        # collecting iteration values
        training_mse.append(train_set_mse)
        test_mse.append(test_set_mse)

        if output:
            print('Adaptive Lasso regression coefficients:', adalasso_model.coef_ / weights)
            print('Training test: MSE:', round(train_set_mse, 4), ', R2:', round(train_set_score, 4))
            print('Test test: MSE:', round(test_set_mse, 4), ', R2:', round(test_set_score, 4))

            # plot that visualizes the coefficients getting shrinked
            ax = plt.gca()
            ax.plot(np.log(LAMBDA_VALUES), coefficients)
            plt.axis('tight')
            plt.xlabel('log(λ)')
            plt.ylabel('Coefficients')
            plt.title('Parameters shrinkage')
            plt.show()

            # plot that visualizes how the objective function of the adaptive lasso changes with lambda
            plt.plot(np.log(LAMBDA_VALUES), obj_function_values)
            plt.axis('tight')
            plt.xlabel('log(λ)')
            plt.ylabel('objective function')
            plt.title('Objective function behavior as λ approaches infinity')
            plt.legend(['λ = ' + str(lambda_value)])
            plt.show()

    ax = plt.gca()
    ax.plot(adalasso_lambdas, training_mse, color = 'red', label = 'Training MSE')
    ax.plot(adalasso_lambdas, test_mse, color = 'lightblue', label = 'Test MSE')
    plt.xlabel('λ')
    plt.ylabel('MSE')
    plt.title('MSE values for different values of λ')
    plt.legend()
    plt.show()

    return train_set_mse, test_set_mse

def testing():
    n_samples, n_features = X_train.shape
    # creating a skeleton for the weights array
    weights = np.ones(n_features)

    # fitting OLS to estimate the weights
    X = sm.add_constant(X_train)
    estimates = sm.OLS(y_train, X).fit()
    # we exlude the intercept since we don't want to add a penalty to it
    idx = 0
    while idx < len(estimates.pvalues)-1:
        if estimates.params[idx+1] < 0.05 and estimates.params[idx+1] > -0.05:
            weights[idx] = sys.maxsize
        else:
            weights[idx] = 1 / np.abs(estimates.params[idx+1])**np.finfo(float).eps
        idx += 1
    print(weights)
    # we apply weights to the covariates
    weighted_X_train = X_train / weights[np.newaxis, :]
    weighted_X_test = X_test / weights[np.newaxis, :]

    # storing the values of the coefficients and the corresponding value of mse for each value of lambda
    lasso = Lasso(max_iter = MAX_ITER)
    coefficients = []
    mse_training = []
    for lambda_value in LAMBDA_VALUES:
        lasso.set_params(alpha = lambda_value)
        lasso.fit(weighted_X_train, y_train)
        # storing the values for each value of lambda for the plots
        coefficients.append(lasso.coef_)
        mse_training.append(mean_squared_error(y_train, lasso.predict(weighted_X_train)))

    # definel model evaluation method
    cv = RepeatedKFold(n_splits = 5, n_repeats = 3, random_state = 1)

    # using cross validation to determine best value for lambda
    cv_lasso = LassoCV(alphas = LAMBDA_VALUES, cv = cv)
    cv_lasso.fit(weighted_X_train, y_train)
    optimal_lambda = cv_lasso.alpha_

    # fitting the lasso regression model with the optimal value of lambda found with cross validation
    lasso_model = Lasso(alpha = 0.4).fit(weighted_X_train, y_train)

    # predicting y values of the training set
    y_train_predicted = lasso_model.predict(weighted_X_train)
    train_set_mse = mean_squared_error(y_train, y_train_predicted)

    # predicting y values of the training set
    y_test_predicted = lasso_model.predict(weighted_X_test)
    test_set_mse = mean_squared_error(y_test, y_test_predicted)
    return

# this method computes elastic net regression using the optimal parameter computed with cross validation
def elastic_net_regression(X_train, X_test, y_train, y_test, output):
    # storing the values of the coefficients and the corresponding value of mse for each value of lambda
    elastic_net = ElasticNet(max_iter = MAX_ITER)
    coefficients = []
    mse_training = []
    for lambda_value in LAMBDA_VALUES:
        elastic_net.set_params(alpha = lambda_value)
        elastic_net.fit(X_train, y_train)
        # storing the values for each value of lambda for the plots
        coefficients.append(elastic_net.coef_)
        mse_training.append(mean_squared_error(y_train, elastic_net.predict(X_train)))

    # definel model evaluation method
    cv = RepeatedKFold(n_splits = 5, n_repeats = 3, random_state = 1)

    # using cross validation to determine best value for lambda
    cv_elastic_net = ElasticNetCV(alphas = LAMBDA_VALUES, cv = cv)
    cv_elastic_net.fit(X_train, y_train)
    optimal_lambda = cv_elastic_net.alpha_
    optimal_l1_ratio = cv_elastic_net.l1_ratio_

    # fitting the lasso regression model with the optimal value of lambda found with cross validation
    elastic_net_model = ElasticNet(alpha = optimal_lambda, l1_ratio = optimal_l1_ratio).fit(X_train, y_train)

    # predicting y values of the training set
    y_train_predicted = elastic_net_model.predict(X_train)
    train_set_mse = mean_squared_error(y_train, y_train_predicted)

    # predicting y values of the training set
    y_test_predicted = elastic_net_model.predict(X_test)
    test_set_mse = mean_squared_error(y_test, y_test_predicted)

    if output:
        # output and analysis
        print('Elastic Net regression coefficients:', elastic_net_model.coef_)
        print('Optimal lambda:', round(optimal_lambda, 4))
        print('Training test: MSE:', round(train_set_mse, 4), ', R2:', round(elastic_net_model.score(X_train, y_train), 4))
        print('Test test: MSE:', round(test_set_mse, 4), ', R2:', round(elastic_net_model.score(X_test, y_test), 4))

        # plot that visualizes the coefficients getting shrinked
        ax = plt.gca()
        ax.plot(np.log(LAMBDA_VALUES), coefficients)
        plt.axis('tight')
        plt.xlabel('log(λ)')
        plt.ylabel('Coefficients')
        plt.title('Parameters shrinkage')
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
