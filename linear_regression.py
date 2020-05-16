import numpy as np
from sklearn.linear_model import LinearRegression

class linearRegression(object):
    def __init__(self,n):
        self.n = n
        self.coef = np.zeros(n)

    def getCoef(self):
        return self.coef

    def train(self, x_train, y_train):
        x_tran = np.transpose(x_train)
        multi = np.dot(x_tran, x_train)
        inv = np.linalg.pinv(multi)
        pseudo_inverse = np.dot(inv, x_tran)
        #print(pseudo_inverse)
        self.coef = np.dot(pseudo_inverse, y_train)
        #print(self.coef)

    def predict(self, x_test):
        y_test = self.coef[0]
        for i in range(1, self.n):
            y_test = y_test + self.coef[i]*x_test[i]
        #print(y_test)
        return y_test
