# libraries import
import sys
import numpy as np
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import cross_validate
from sklearn.linear_model import ElasticNet

#
def compute_elastic_net_regression(X_train, X_test, y_train, y_test):
    # fitting elastic net regression with cross validation to obtain the best value of lambda

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
