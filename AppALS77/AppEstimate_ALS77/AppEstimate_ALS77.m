function [theta, ster, logMLE] = AppEstimate_ALS77()

global y x1 x2 x3 p1 p2 p3;


alpha=-11;
beta1=.03;
beta2=1.1;
beta3=-.01;
sigma2u=0.01;
sigma2v=0.0003;
lsigma2u = log(sigma2u);
lsigma2v = log(sigma2v);


    
    theta0 = [alpha beta1 beta2 beta3 lsigma2u lsigma2v]'; % 
    Options = optimset('TolX', 1e-10, 'TolFun', 1e-10, ...
                    'MaxIter', 20000, ...   
                    'MaxFunEvals', 6 * 20000,...
                    'Display', 'iter-detailed');
    
         

    [theta, logMLE, ~, ~, ~, hessian] = fminunc(@AppLoglikelihood_ALS77, theta0, Options);%
    %[theta, logMLE, ECMLE, output] = fminsearch(@AppLoglikelihood_ALS77, theta0, Options);%
    
    %% standard errors
    theta(5:6) =  exp(theta(5:6));    
    delta = 1e-6;
     grad = zeros(length(y), length(theta));
     %add = delta*ones(1, length(theta));
     for i=1:length(theta)
         theta1 = theta;
         theta1(i) = theta(i) + delta;
         grad(:,i) = (AppLogDen_ALS77(theta1)-AppLogDen_ALS77(theta))/delta; 
     end
    OPG = grad'*grad; 
    
    D = diag([1,1,1,1,theta(5:6)']); % Delta method

    [theta sqrt(diag(inv(OPG)))];% sqrt(diag(D'*inv(hessian)*D))   sqrt(diag(D'*inv(hessian)*D*OPG*D'*inv(hessian)*D))]

    ster = sqrt(diag(inv(OPG)));  
    
          
end

