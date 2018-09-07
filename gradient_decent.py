import pandas as pd
from ex1 import sigmoid as sig
import matplotlib.pyplot as plt
import numpy as np

data = pd.read_csv("A4_Q7_data.csv")
X = np.array(data['X'])
Y = np.array(data['Y'])


def loss(y,yr):
    return 0.5 *(y-yr)**2

def error(w,b):
    err = 0.0 
    for x,y in zip(X,Y):
        fx = sig((w*x+b))
        err += loss(fx,y)
    return err

def grad_b(w,b,x,y):
    fx = sig((w*x+b))
    return (fx - y) * fx *(1-fx)
def grad_w(w,b,x,y):
    fx = sig((w*x+b))
    return (fx - y) * fx *(1-fx) * x


def do_gradient_decent():
    w,b,eta,max_epochs,dw,db=1,1,0.01,100,0,0
    w_history,b_history,error_history = [],[],[]
    beta1,beta2=0.9,0.99
    m_w,m_b,v_w,v_b,eps=0,0,0,0,1e-8

        

    for i in range(max_epochs):
        dw, db =0,0

        for x,y in zip(X,Y):
            dw +=grad_w(w,b,x,y)
            db +=grad_b(w,b,x,y)
        #adam loss function
        m_w = beta1 * m_w +(1-beta1)*dw
        m_b = beta1 * m_b +(1-beta1)*db

        v_w = beta2 * v_w +(1-beta2)*dw**2
        v_b = beta2 * v_b +(1-beta2)*db**2

        w = w - (eta / np.sqrt(v_w + eps))*m_w
        b  = b - (eta / np.sqrt(v_b + eps)) * m_b

        w_history.append(w)
        b_history.append(b)
        error_history.append(error(w,b))
    from mpl_toolkits.mplot3d import axes3d
    fig = plt.figure()
    ax = fig.gca(projection='3d')

    # Plot a basic wireframe.
    from matplotlib import cm

    surf = ax.plot_surface(w_history, b_history, error_history,cmap=cm.coolwarm,
                       linewidth=0, antialiased=False)
    print(error_history)
    fig.colorbar(surf, shrink=0.5, aspect=5)
    plt.contour(w_history, b_history, error_history)
    plt.show()
        
    
do_gradient_decent()    
    
    






