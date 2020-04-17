# Electricity Generators data set from SL79-80
# Corrected OLS for deterministic frontier model

import numpy                   as np
import pandas                  as pd
import matplotlib.pyplot       as mp
import statsmodels.formula.api as sm

from sklearn import linear_model


#~~~~~~  1. Import data  ~~~~~~#

SOURCE = "../DataSets/cowing.xlsx"
df = pd.read_excel(SOURCE)

for col in ["P1", "P2", "P3"]:
    df[col] = np.log(df[col])
    

#~~~~~~  2. Estimation   ~~~~~~#

pred_model = linear_model.LinearRegression()
pred_model.fit(df[["X1", "X2", "X3"]], df["y"])
pred       = pred_model.predict(df[["X1", "X2", "X3"]])

outp_model = sm.ols(formula = 'y ~ X1 + X2 + X3', data=df[["y", "X1", "X2", "X3"]])
fitted     = outp_model.fit()

resid     = df["y"] - pred
u_star    = -(resid - resid.max())
eff_colsd = np.exp(-u_star)


#~~~~~~  3. Output       ~~~~~~#

print("colsd = \nLinear regression model:\n\ty ~ X1 + X2 + X3")
print("\nEstimated Coefficients:")
print(fitted.summary())

mp.hist(eff_colsd);
