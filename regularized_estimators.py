import pandas as pd
import matplotlib.pyplot as plt

# preprocessing libraries
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

# regression methods computations
from regression_methods import linear_regression, ridge_regression, lasso_regression, adaptive_lasso_regression, elastic_net_regression

# data import
data = pd.read_csv('housing.csv', header = None)
colnames = ['CRIM', 'ZN', 'INDUS', 'CHAS', 'NOX', 'RM', 'AGE', 'DIS', 'RAD', 'TAX', 'PTRATIO', 'B', 'LSTAT', 'MEDV']
data.columns = colnames

# separating the dataset into covariates and response
X = data.drop(['MEDV'], axis = 1)
y = data['MEDV']

outputs = True
plots = False

# separating the dataset into training set and test set (with seed)
X_train, X_test, y_train, y_test = train_test_split(X, y, train_size = 0.75, random_state = 123)

# scaling the covariates in order to make them uniform
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)


# applying Linear Regression model and obtaining mse on train and test set
print("Computing Linear Regression...")
training_linear_mse, test_linear_mse = linear_regression(X_train, X_test, y_train, y_test, outputs)

# applying Ridge Regression model with optimal lambda and obtaining mse on train and test set
print("\nComputing Ridge Regression with cross-validation...")
training_ridge_mse, test_ridge_mse = ridge_regression(X_train, X_test, y_train, y_test, outputs, plots)

# applying Lasso regression model with optimal lambda and obtaining mse on train and test set
print("\nComputing Lasso with cross-validation...")
training_lasso_mse, test_lasso_mse = lasso_regression(X_train, X_test, y_train, y_test, outputs, plots)

# applying Adaptive lasso regression model and obtaining mse on train and test set
print("\nComputing Adaptive Lasso...")
training_adalasso_mse, test_adalasso_mse = adaptive_lasso_regression(X_train, X_test, y_train, y_test, outputs, plots)

# applying Elastic Net regression model with optimal lambda and obtaining mse on train and test set
print("\nComputing Elastic Net with cross-validation...")
training_elastic_net_mse, test_elastic_net_mse = elastic_net_regression(X_train, X_test, y_train, y_test, outputs, plots)


#
