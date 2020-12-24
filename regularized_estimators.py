import pandas as pd

# preprocessing libraries
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

# regression methods computations
from linear_regression import compute_linear_regression
from ridge_regression import compute_best_lambda_ridge, compute_ridge_regression


# data import
data = pd.read_csv('housing.csv', header = None)
colnames = ['CRIM', 'ZN', 'INDUS', 'CHAS', 'NOX', 'RM', 'AGE', 'DIS', 'RAD', 'TAX', 'PTRATIO', 'B', 'LSTAT', 'MEDV']
data.columns = colnames

# separating the dataset into covariates and response
X = data.drop(['MEDV'], axis = 1)
y = data['MEDV']

# separating the dataset into training set and test set (with seed)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.25, random_state = 1)

# scaling the covariates in order to make them uniform
scaler = StandardScaler().fit(X_train)
X_train = scaler.transform(X_train)
X_test = scaler.transform(X_test)

# applying linear regression model and obtaining mse on train and test set
training_set_mse_LR, test_set_mse_LR = compute_linear_regression(X_train, X_test, y_train, y_test)

# applying ridge regression model with optimal lambda and obtaining mse on train and test set
optimal_lambda = compute_best_lambda_ridge(X_train, y_train)
training_set_mse_RR, test_set_mse_RR = compute_ridge_regression(X_train, X_test, y_train, y_test, optimal_lambda)

print(training_set_mse_LR, test_set_mse_LR)
print(training_set_mse_RR, test_set_mse_RR)

#
