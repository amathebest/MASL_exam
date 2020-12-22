# regression libraries
from sklearn.metrics import mean_square_error
from sklearn.linear_model import LinearRegression

def compute_linear_regression(X_train, X_test, y_train, t_test):
    # applying linear regression to the training set
    lin_reg = LinearRegression().fit(X_train, y_train)

    # predicting y values of the training set
    y_train_predicted = lin_reg.predict(X_train)
    train_set_mse = mean_square_error(y_train, y_train_predicted)
    


    return train_set_mse, test_set_mse
