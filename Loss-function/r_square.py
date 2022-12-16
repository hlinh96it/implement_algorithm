import numpy as np
from sklearn.datasets import load_boston
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score

def cal_r2_score(y_true, y_pred):
    y_bar = np.mean(y_true)
    ss_total = np.sum((y_true - y_bar) ** 2)
    ss_explained = np.sum((y_pred - y_bar) ** 2)
    ss_residual = np.sum((y_true - y_pred) ** 2)
    sklearn_r2 = r2_score(y_true, y_pred)
    
    print(f'R_square 1 - (SS_residual / SS_total) = {1 - ss_residual/ss_total}')
    print(f'R-square sklearn {sklearn_r2}')


X = load_boston()['data'].copy()
y = load_boston()['target'].copy()
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
model = LinearRegression()
model.fit(X_train, y_train)

predictions = model.predict(X_test)
cal_r2_score(y_test, predictions)
