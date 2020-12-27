import pandas as pd

# preprocessing libraries
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

# regression methods computations
from regression_methods import compute_linear_regression, compute_ridge_regression
from lasso_regression import compute_lasso_regression
from elastic_net_regression import compute_elastic_net_regression

# data import
data = pd.read_csv('housing.csv', header = None)
colnames = ['CRIM', 'ZN', 'INDUS', 'CHAS', 'NOX', 'RM', 'AGE', 'DIS', 'RAD', 'TAX', 'PTRATIO', 'B', 'LSTAT', 'MEDV']
data.columns = colnames

# separating the dataset into covariates and response
X = data.drop(['MEDV'], axis = 1)
y = data['MEDV']

# separating the dataset into training set and test set (with seed)
X_train, X_test, y_train, y_test = train_test_split(X, y, train_size = 2*(1/3), random_state = 0)

# scaling the covariates in order to make them uniform
scaler = StandardScaler().fit(X_train)
X_train = scaler.transform(X_train)
X_test = scaler.transform(X_test)

if True:
    outputs = True

    # applying linear regression model and obtaining mse on train and test set
    print("Computing Linear Regression...")
    training_linear_mse, test_linear_mse = compute_linear_regression(X_train, X_test, y_train, y_test, True)

    # applying ridge regression model with optimal lambda and obtaining mse on train and test set
    print("\nComputing Ridge Regression with cross-validation...")
    training_ridge_mse, test_ridge_mse = compute_ridge_regression(X_train, X_test, y_train, y_test, outputs)


else:


    # applying lasso regression model with optimal lambda and obtaining mse on train and test set
    print("\nComputing Lasso with cross-validation...")
    training_lasso_mse, test_lasso_mse = compute_lasso_regression(X_train, X_test, y_train, y_test)


    # applying elastic net regression model with optimal lambda and obtaining mse on train and test set
    print("Computing Elastic Net with cross-validation...")
    training_elastic_net_mse, test_elastic_net_mse = compute_elastic_net_regression(X_train, X_test, y_train, y_test)



#
