import pandas as pd

# preprocessing libraries
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

# regression methods computations
from linear_regression import compute_linear_regression
from ridge_regression import compute_ridge_regression
from lasso_regression import compute_lasso_regression

# data import
data = pd.read_csv('housing.csv', header = None)
colnames = ['CRIM', 'ZN', 'INDUS', 'CHAS', 'NOX', 'RM', 'AGE', 'DIS', 'RAD', 'TAX', 'PTRATIO', 'B', 'LSTAT', 'MEDV']
data.columns = colnames

# separating the dataset into covariates and response
X = data.drop(['MEDV'], axis = 1)
y = data['MEDV']

# separating the dataset into training set and test set (with seed)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 2*(1/3), random_state = 1)

# scaling the covariates in order to make them uniform
#scaler = StandardScaler().fit(X_train)
#X_train = scaler.transform(X_train)
#X_test = scaler.transform(X_test)

# applying linear regression model and obtaining mse on train and test set
print("Computing Linear Regression...")
training_linear_mse, test_linear_mse = compute_linear_regression(X_train, X_test, y_train, y_test)
print(training_linear_mse, test_linear_mse)

# applying ridge regression model with optimal lambda and obtaining mse on train and test set
print("Computing Ridge Regression with cross-validation...")
training_ridge_mse, test_ridge_mse = compute_ridge_regression(X_train, X_test, y_train, y_test)
print(training_ridge_mse, test_ridge_mse)

# applying lasso regression model with optimal lambda and obtaining mse on train and test set
print("Computing Lasso with cross-validation...")
training_lasso_mse, test_lasso_mse = compute_lasso_regression(X_train, X_test, y_train, y_test)
print(training_lasso_mse, test_lasso_mse)


#
