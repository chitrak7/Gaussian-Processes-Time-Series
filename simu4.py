#Simulation with local trend 

import numpy as np
import matplotlib.pyplot as plt
from scipy import signal
f=np.pi/25
h=1
k=50
k1=25
p=3
beta0 = np.array([1,0.005,0.0005])
n=500
n_tr = 300
n_te = n - n_tr

def st(t):
    return 10*signal.sawtooth(f*t[1])

def m(t):
    return np.dot(t,beta0)
    #return 0

def sh(i,j):
    return np.abs(np.sin(np.pi*(i-j)*(1/k)))

def df(i,j):
    return np.abs(i-j)/k1

def mc(r):
    return (h*h)*np.exp(-5*(r**2))

def ker(i,j):
    #print(i,j)
    return mc(sh(i,j)) + mc(df(i,j))

sd=1
x = np.arange(n,dtype=int)
x_tr = x[:n_tr]
x_te = np.setdiff1d(x,x_tr)
T = np.array([[pow(i,j) for j in range(p)]for i in x])
Train = np.array([[pow(i,j) for j in range(p)]for i in x_tr])
Test = np.array([[pow(i,j) for j in range(p)]for i in x_te])
y = np.array([st(i)+m(i) for i in T])
y_tr  = [i + sd*np.random.standard_normal(1) for i in y[x_tr]]
y_te  = np.zeros(shape=(n_te))


aa = np.arange(n)
C = np.array([[ker(i,j) for j in x_tr] for i in x_tr]) + np.matmul(Train,Train.T)
C = np.linalg.inv(C + 0.5*np.identity(n_tr)) 
beta = np.linalg.inv(np.matmul(Train.T,np.matmul(C,Train)))
beta1 = np.matmul(Train.T,np.matmul(C,y_tr))
beta  = np.matmul(beta,beta1)
trend_c = np.matmul(T,beta)
mu = y_tr 
mu1 = np.matmul(Test,beta)
mu2 = np.matmul(T,beta)
print(beta)
for i in range(len(x_te)):
    k_i = np.array([ker(x_te[i],j) + np.dot(T[j],T[x_te[i]]) for j in x_tr])
    y_te[i] = np.dot(np.matmul(C,k_i),mu)

sd_l = np.zeros(shape=(n))
sd_u = np.zeros(shape=(n))

# for i in range(len(x)):
#     k_i = np.array([ker(x[i],j) + np.dot(T[j],T[i]) for j in x_tr])
#     var = sd**2 + ker(x[i],x[i]) + np.dot(T[i],T[i]) - np.dot(np.matmul(C,k_i),k_i)
#     ee = np.dot(np.matmul(C,k_i),mu)
#     sd1 = 1.96*np.sqrt(var)
#     sd_l[i] = ee-sd1
#     sd_u[i] = ee+sd1


plt.plot(x,y,label="actual curve")
plt.fill_between(x,sd_l,sd_u,color="powderblue",label="95%% confidance interval")
plt.plot(x_tr,y_tr,'.',label='train')
plt.plot(x,trend_c,label='trend')
plt.plot(x_te,y_te,'.',label='test')
plt.legend()
plt.show()

