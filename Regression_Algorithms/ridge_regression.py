import numpy as np

from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.linear_model import Ridge, ridge_regression

from sklearn.datasets import load_diabetes

import matplotlib.pyplot as plt
plt.style.use('default')
plt.rcParams['axes.grid'] = True
plt.rcParams["figure.figsize"] = (16, 3)
plt.rcParams['figure.dpi'] = 150

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression

import warnings
warnings.filterwarnings("ignore")


#%% Loading data
X, y = load_diabetes(return_X_y=True)  # return_X_y return (data, target)
features = load_diabetes()['feature_names']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)


#%% Adjust the value of alpha to see with difference alpha, how the fitting line of the model is
n_alphas = 100
alphas = 1 / np.logspace(1, -2, n_alphas)
coefs = []

for alpha in alphas:
    ridge = Ridge(alpha, fit_intercept=False)
    ridge.fit(X_train, y_train)
    coefs.append(ridge.coef_)


#%% See model output for different alpha level
plt.plot(alphas, coefs)
plt.xscale('log')
plt.xlabel('Alpha level')
plt.ylabel('Coef of features')
plt.title('Ridge coefficients when alpha change', fontsize=15)
plt.legend(features)
plt.axis('tight')
plt.show()