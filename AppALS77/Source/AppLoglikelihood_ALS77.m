function logL = AppLoglikelihood_Copula(coefs)

%% tranform parameters back true range

     coefs(5:6) =  exp(coefs(5:6));

%% obtain the log likelihood

    logDen = AppLogDen_ALS77(coefs);
    logL = -sum(logDen);

end 