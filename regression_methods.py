# libraries import
import sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import cross_val_score
from sklearn.metrics import mean_squared_error
from sklearn.linear_model import Ridge, LinearRegression

# this method computes linear regression 
def compute_linear_regression(X_train, X_test, y_train, y_test, outputs = False):
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
        print('Linear regression coefficients:', lin_reg.coef_)
        print('Training test: MSE:', round(train_set_mse, 4), ', R2:', round(lin_reg.score(X_train, y_train), 4))
        print('Test test: MSE:', round(test_set_mse, 4), ', R2:', round(lin_reg.score(X_test, y_test), 4))

    return train_set_mse, test_set_mse

# this method computes ridge regression using the optimal parameter computed with cross validation
def compute_ridge_regression(X_train, X_test, y_train, y_test, outputs = False):
    # pool of possible candidates for lambda value
    lambda_values = np.arange(0.0001, 1000, 0.1)

    # using cross validation to determine best value for lambda
    scores = []
    for lambda_value in lambda_values:
        ridge_model_cv = Ridge(alpha = lambda_value)
        mse_lambda_value = cross_val_score(ridge_model_cv, X_train, y_train, cv = 5, scoring = 'neg_mean_squared_error', n_jobs = -1)
        scores.append(-np.mean(mse_lambda_value))

    # storing the values of lambda and their corresponding score into a dataframe
    lambda_scores_df = pd.DataFrame({'log_lambda': np.log(lambda_values), 'scores': scores})

    # computing the best value of lambda, e.g. the one with minimum MSE
    min_score = lambda_scores_df.loc[lambda_scores_df['scores'].idxmin()]['log_lambda']

    # fitting the ridge regression model with the optimal value of lambda found with cross validation (minimum score value)
    ridge_model = Ridge(alpha = np.e**min_score).fit(X_train, y_train)

    # predicting y values of the training set
    y_train_predicted = ridge_model.predict(X_train)
    train_set_mse = mean_squared_error(y_train, y_train_predicted)

    # predicting y values of the training set
    y_test_predicted = ridge_model.predict(X_test)
    test_set_mse = mean_squared_error(y_test, y_test_predicted)

    # output and analysis
    if outputs:
        print('Ridge regression coefficients:', ridge_model.coef_)
        print('Optimal lambda:', np.e**min_score)
        print('Training test: MSE:', round(train_set_mse, 4), ', R2:', round(ridge_model.score(X_train, y_train), 4))
        print('Test test: MSE:', round(test_set_mse, 4), ', R2:', round(ridge_model.score(X_test, y_test), 4))

        # plot for cross validation
        plt.scatter(lambda_scores_df['log_lambda'], lambda_scores_df['scores'], s = 1, color = 'blue')
        plt.vlines(x = min_score, ymin = np.min(lambda_scores_df['scores']), ymax = np.max(lambda_scores_df['scores']), color = 'red', zorder = 2)
        plt.xlabel('log(λ)')
        plt.ylabel('MSE')
        plt.title('Optimal value of λ using cross-validation')
        plt.show()

    return train_set_mse, test_set_mse
