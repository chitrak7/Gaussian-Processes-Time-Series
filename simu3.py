#Simulation with trend and seasonality

import numpy as np
import matplotlib.pyplot as plt

f=np.pi/25
h=1
k=50
p=3
beta0 = np.array([1,0.02,0.000])

def st(t):
    return 10*np.sin(f*t[1])

def m(t):
    return np.dot(t,beta0)
    #return 0

def sh(i,j):
    return np.abs(np.sin(np.pi*(i-j)*(1/k)))

def df(i,j):
    return np.abs(i-j)/k

def mc(r):
    return (h*h)*np.exp(-5*(r**2))

def ker(i,j):
    #print(i,j)
    return mc(sh(i,j)) + mc(df(i,j))

sd=0.5
x = np.arange(500)
x_tr = x[:400]
x_te = x[400:]
T = np.array([[pow(i,j) for j in range(p)]for i in x])
Train = np.array([[pow(i,j) for j in range(p)]for i in x_tr])
Test = np.array([[pow(i,j) for j in range(p)]for i in x_te])
y = [st(i)+m(i) for i in T]
y_tr  = [i + sd*np.random.standard_normal(1) for i in y[:400]]
y_te  = np.zeros(shape=(100))


aa = np.arange(500)
C = np.array([[ker(i,j) for j in aa[0:400]]for i in aa[0:400]])
C = np.linalg.inv(C + 0.5*np.identity(400)) 
beta = np.linalg.inv(np.matmul(Train.T,np.matmul(C,Train)))
beta1 = np.matmul(Train.T,np.matmul(C,y_tr))
beta  = np.matmul(beta,beta1)
trend_c = np.matmul(T,beta)
mu = y_tr - np.matmul(Train,beta)
mu1 = np.matmul(Test,beta)
print(beta)
for i in range(100):
    k_i = np.array([ker(x_te[i],j) for j in x_tr])
    y_te[i] = mu1[i] + np.dot(np.matmul(C,k_i),mu)
plt.plot(x,y,label="actual curve")
plt.plot(x_tr,y_tr,'.',label='train')
plt.plot(x,trend_c,label='trend')
plt.plot(x_te,y_te,'.',label='test')
plt.legend()
plt.show()

