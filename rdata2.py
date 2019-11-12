#Simulation with local trend 

import numpy as np
import matplotlib.pyplot as plt
from scipy import signal
f=np.pi/25
h=0.5
p=1
beta0 = np.array([1,0.02,0.00004,0.00000])


def st(t):
    return 10*np.sin(f*t[1])

def m(t):
    return np.dot(t,beta0)
    #return 0

def sh(i,j,k):
    return np.abs(np.sin(np.pi*(i-j)*(1/k)))

def df(i,j,k1):
    return 2*np.abs(i-j)/k1

def mc(r):
    return (h*h)*np.exp(-1*(r**2))

def ker(i,j,k,k1):
    #print(i,j)
    return mc(sh(i,j,k)) + mc(df(i,j,k1))

sd=0.1
a = np.load("tide1_data.npy",allow_pickle=True)
x = np.arange(a.shape[0])
n=a.shape[0]
n_tr = n-60
n_te = n-n_tr
x_tr = x[:n_tr]
x_te = x[n_tr:]
T = np.array([[pow(i,j) for j in range(p)]for i in x])
Train = np.array([[pow(i,j) for j in range(p)]for i in x_tr])
Test = np.array([[pow(i,j) for j in range(p)]for i in x_te])
y = a
y_tr  = y[x_tr]
y_te  = np.zeros(shape=(n_te))

k_seq = np.arange(10,70,step=2)
cve = np.zeros(shape=(len(k_seq)))
n_cvtr = 100
n_cvte = n_tr - n_cvtr
cv_tr = x_tr[:n_cvtr]
cv_te = x_tr[n_cvtr:]
cvy_tr = y_tr[:n_cvtr]
cvy_te = y_tr[n_cvtr:]
cv_Train = np.array([[pow(i,j) for j in range(p)]for i in cv_tr])
cv_Test = np.array([[pow(i,j) for j in range(p)]for i in cv_te])
for i in range(len(k_seq)):
    k=k_seq[i]
    C = np.array([[ker(i,j,k,k) for j in cv_tr] for i in cv_tr])
    C = np.linalg.inv(C + 0.5*np.identity(n_cvtr)) 
    beta = np.linalg.inv(np.matmul(cv_Train.T,np.matmul(C,cv_Train)))
    beta1 = np.matmul(cv_Train.T,np.matmul(C,cvy_tr))
    beta  = np.matmul(beta,beta1)
    cv = np.zeros(shape=(n_cvte))
    mu = cvy_tr - np.matmul(cv_Train,beta)
    mu1 = np.matmul(cv_Test,beta)
    for j1 in range(len(cv_te)):
        k_i = np.array([ker(cv_te[j1],j,k,k) for j in cv_tr])
        cv[j1] = mu1[j1] + np.dot(np.matmul(C,k_i),mu)
    delta = np.array([cv[i]-cvy_te[i] for i in range(n_cvte)]).flatten()
    cve[i] = np.dot(delta,delta)
    print(k_seq[i],cve[i])

plt.plot(k_seq,cve)
plt.show()
plt.clf()

k = k_seq[np.argmin(cve)]
k1 = k
k=58
# k1=k
aa = np.arange(n)
C = np.array([[ker(i,j,k,k1) for j in x_tr] for i in x_tr])
C = np.linalg.inv(C + 0.5*np.identity(n_tr)) 
print(Train.shape)
beta = np.linalg.inv(np.matmul(Train.T,np.matmul(C,Train)))
beta1 = np.matmul(Train.T,np.matmul(C,y_tr))
beta  = np.matmul(beta,beta1)
trend_c = np.matmul(T,beta)
mu = y_tr - np.matmul(Train,beta)
mu1 = np.matmul(Test,beta)
mu2 = np.matmul(T,beta)
print(beta)
for i in range(len(x_te)):
    k_i = np.array([ker(x_te[i],j,k,k1) for j in x_tr])
    y_te[i] = mu1[i] + np.dot(np.matmul(C,k_i),mu)

sd_l = np.zeros(shape=(n))
sd_u = np.zeros(shape=(n))

for i in range(len(x)):
    k_i = np.array([ker(x[i],j,k,k1) for j in x_tr])
    var = sd**2 + ker(x[i],x[i],k,k1) - np.dot(np.matmul(C,k_i),k_i)
    mu2[i] = mu2[i] + np.dot(np.matmul(C,k_i),mu)
    ee = mu2[i]
    sd1 = 1.96*np.sqrt(var)
    sd_l[i] = ee-sd1
    sd_u[i] = ee+sd1


plt.plot(x,y,label="actual curve")
plt.fill_between(x,sd_l,sd_u,color="powderblue",label="95%% confidance interval")
#plt.plot(x_tr,y_tr,'.',label='train')
plt.plot(x,mu2,label='prediction')
plt.plot(x_te,y_te,label='test')
#plt.plot(x_te,y[x_te],'.',label='real')
plt.legend()
plt.show()
