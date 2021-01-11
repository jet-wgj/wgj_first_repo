import numpy as np

class OLSlinearRegression:

    def _ols(self,X,y):
        '''calculate w with data X and y'''
        tmp = np.linalg.inv(np.matmul(X.T,X))
        tmp = np.matmul(tmp,X.T)
        return np.matmul(tmp,y)

    def data_process_X(self,X_data):
        '''process the training data X'''
        m,n=X_data.shape
        X_pro=np.empty((m,n+1))
        X_pro[:,0]=1
        X_pro[:,1:]=X_data

        return X_pro

    def train(self,X_train,y_train):
        '''train the data'''
        X_train=self.data_process_X(X_train)
        self.w=self._ols(X_train,y_train)
        return None

    def predict(self,X_new):
        '''predict new y's with the X_new(a matrix of new data like X_train)'''
        '''the function returns a vector whose members are predicting values'''
        X_new=self.data_process_X(X_new)
        return np.matmul(X_new,self.w)

class GDLinearRegression:

    def __init__(self,eta=1e-3,n_iter=200,tol=None):
        '''the construct function.eta is the argment multiplied with the gradient;'''
        '''n_iter means the largest iterating times;tol is for stop early'''
        self.eta=eta
        self.n_iter=n_iter
        self.tol=tol
        self.w=None

    def _loss(self,w,X,y):
        '''calculate the value of loss function'''
        return np.sum((np.matmul(X,w)-y)**2)/y.size

    def _gradient(self,w,X,y):
        '''calculate the gradient of the target w'''
        return np.matmul(X.T,(np.matmul(X,w)-y))/y.size

    def _gradient_descent(self,w,X,y):
        '''realize the gradient descent algorithm'''
        loss_old=np.inf
        for i in range(self.n_iter):
            loss=self._loss(w,X,y)
            print('%4i Loss: %s' % (i,loss))
            tmp=loss_old-loss
            if self.tol is not None:
                if tmp<self.tol:
                    break
            else:
                if tmp==0:
                    break

            loss_old=loss
            w-=self.eta*self._gradient(w,X,y)

        return w

    def _data_process(self,X_data):
        '''process the training data'''
        m,n=X_data.shape

        X_=np.empty((m,n+1))
        X_[:,0]=1
        X_[:,1:]=X_data
        return X_

    def train(self,X_data,y_data):
        '''train the data,and get the arg w'''
        X=self._data_process(X_data)
        _,n=X.shape
        self.w=np.random.random(n)*0.05

        self.w=self._gradient_descent(self.w,X,y_data)

    def predict(self,X_new):
        '''predict the values of X_new'''
        return np.matmul(self._data_process(X_new),self.w)

