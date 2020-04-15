% Electricity Generators data set from SL79-80
% Corrected OLS for deterministic frontier model

%warning ('off', clear)

%% 1. Import data

data=xlsread('~/Documents/Matlab/DataSets/cowing.xlsx');

global y x1 x2 x3 p1 p2 p3 

y=data(:,2); % output and inputs are presumably in logs
x1=data(:,3);
x2=data(:,4);
x3=data(:,5);
p1 = log(data(:,6)); % original prices are NOT logs
p2 = log(data(:,7));
p3 = log(data(:,8));

%% Estimation 

colsd = LinearModel.fit([x1 x2 x3], y)
resid = colsd.Residuals.Raw;
u_star = - (resid - max(resid));
eff_colsd = exp(-u_star);
hist(eff_colsd);


