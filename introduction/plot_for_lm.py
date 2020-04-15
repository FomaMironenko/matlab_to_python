import numpy      as np
import pandas     as pd
import matplotlib.pyplot as mp

from sklearn import linear_model
import math

SOURCE = "../DataSets/cowing.xlsx"


def main():
	df = pd.read_excel(SOURCE)

	for col in ["P1", "P2", "P3"]:
    		df[col] = df[col].apply(lambda x: math.log(x))
    
	model = linear_model.LinearRegression()
	model.fit( df[["X1", "X2", "X3"]], df["y"] )
	
	resid     = df["y"] - model.predict(df[["X1", "X2", "X3"]])
	u_star    = -(resid - resid.max())
	eff_colsd = u_star.apply(lambda x: math.exp(-x))
	
	mp.hist(eff_colsd)


if __name__ == "__main__":
	main()



