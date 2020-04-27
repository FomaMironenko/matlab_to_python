### Electricity generators data set from SL79-80
### MLE ALS77

import numpy              as np
import pandas             as pd
import matplotlib.pyplot  as mp
from   scipy.stats    import norm
from   scipy.optimize import minimize, OptimizeResult
import sys


#%%~~~~ 1. import data ~~~~%%#

SOURCE = "../DataSets/cowing.xlsx"

df = pd.read_excel(SOURCE)
y = df["y"]
[x1, x2, x3] = [df["X1"], df["X2"], df["X3"]]
[p1, p2, p3] = [np.log(df["P1"]), np.log(df["P2"]), np.log(df["P3"])]



#%%~~~~ 2. functions implementation ~~~~%%#

def AppLogDen_ALS77(Pars: np.array):
    
    global y, x1, x2, x3, p1, p2, p3
    
    [alpha, beta1, beta2, beta3, sigma2u, sigma2v] = Pars[0:6]
    
    Lambda = np.sqrt( sigma2u / sigma2v )
    sigma2 = sigma2u + sigma2v
    sigma  = np.sqrt( sigma2 )
    
    eps = y  - alpha - x1*beta1 - x2*beta2 - x3*beta3
    
    ## in norm.pdf and norm.cdf loc=0 and scale=1 by default 
    Den = (2/sigma) * norm.pdf(eps / sigma) * norm.cdf(-Lambda * eps / sigma)
    
    return np.log(Den)


def AppLoglikelihood_ALS77(coefs: np.array):
    AppLoglikelihood_ALS77.calls += 1
    
    ## transform parametrs back true range
    coefs[4:6] = np.exp(coefs[4:6])
    
    ## obtain the log likelihood
    logDen = AppLogDen_ALS77(coefs)
    return -sum(logDen)
AppLoglikelihood_ALS77.calls = 0
AppLoglikelihood_ALS77.it    = 0
AppLoglikelihood_ALS77.prev  = []


def _callback(vec):
    App = AppLoglikelihood_ALS77
    App.calls -= 1
    print("\r" + " "*100 + "\r", end = "")
    print("| %d\t\t | %d\t\t | %.4f\t | %.1e\t|" % (
          App.it, App.calls, App(np.copy(vec)),
          np.linalg.norm(App.prev - vec)           ), 
          end = "")
    
    App.it  += 1 
    App.prev = np.copy(vec)
    
def AppEstimate_ALS77():
    
    global y, x1, x2, x3, p1, p2, p3, it
    
    ## starting point
    alpha    = -11 
    beta1    = 0.03
    beta2    = 1.1
    beta3    = -0.01
    
    sigma2u  = 0.01
    sigma2v  = 0.0003
    lsigma2u = np.log(sigma2u)
    lsigma2v = np.log(sigma2v)
    
    theta0 = np.array([alpha, beta1, beta2, beta3, lsigma2u, lsigma2v])
    
    ## estimation
    print("| Iteration\t | F-count \t | f(x)\t\t | Step-size\t|")

    AppLoglikelihood_ALS77.it    = 1
    AppLoglikelihood_ALS77.calls = 1
    AppLoglikelihood_ALS77.prev  = np.copy(theta0)
    
    rez = minimize(AppLoglikelihood_ALS77, theta0,
                   method="Powell",
                   options={"disp": False,
                            "xtol": 1e-10,
                            "ftol": 1e-10,
                            "maxiter": 20000},
                   callback=_callback);
    
    theta  = rez.x
    logMLE = rez.fun
    
    ## standard errors
    theta[4:6] = np.exp(theta[4:6])
    delta = 1e-6
    grad  = pd.DataFrame( np.zeros((len(y), len(theta))) )
    
    for i in range(len(theta)):
        theta1 = np.copy(theta)
        theta1[i] += delta
        grad.iloc[:, i] = (AppLogDen_ALS77(theta1) - 
                           AppLogDen_ALS77(theta )) / delta
        
    OPG  = grad.transpose().dot(grad)
    D    = np.diag( np.concatenate(([1, 1, 1, 1], theta[4:6])) )
    ster = np.sqrt( np.diag(np.linalg.inv(OPG)) )
    
    return [theta, ster, logMLE]



#%%~~~~ 3. Maximum Likelihood estimation ~~~~%%#
sys.stderr = None 

[coefs, ster, logMLE] = AppEstimate_ALS77()

print("\n\nans =")
for i in range(len(coefs)):
    print("\t%.4f" % coefs[i])
    
sys.stderr = sys.__stderr__



#%%~~~~ 4. Prediction of inefficiencies ~~~~%%#
    
eps = y - coefs[0] - x1*coefs[1] - x2*coefs[2] - x3*coefs[3]

sig = np.sqrt(coefs[4] + coefs[5])
lam = np.sqrt(coefs[4] / coefs[5])

bi  = eps * lam / sig
haz = norm.pdf(bi) / (1 - norm.cdf(bi))
sigstar = np.sqrt(coefs[4] * coefs[5]) / sig

Eu = sigstar * (haz - bi)
Vu = (sigstar**2) * (1 + bi*haz - haz*haz)

mp.hist(Eu);



#%%~~~~ 5. Output ~~~~%%#

output = pd.DataFrame(index  =["alpha", "beta1", "beta2", "beta3", "sig2u", "sig2v"], 
                      columns=["Est", "stErr", "t-stat", "p-val", "95%conf", "Interv"])
output["Est"]     = coefs
output["stErr"]   = ster
output["t-stat"]  = coefs / ster
output["p-val"]   = 2 * (1 - norm.cdf(np.absolute(output["t-stat"])) )
output["95%conf"] = norm.ppf(0.025, coefs, ster)
output["Interv"]  = norm.ppf(0.975, coefs, ster)

print("LL %.2f" % logMLE)
print(output)

