function logDen = AppLogDen_ALS77(Pars)
     global y;
     global x1;
     global x2;
     global x3;
     global p1;
     global p2;
     global p3;

%     n=length(y); 
     alpha=Pars(1);
     beta1=Pars(2);
     beta2=Pars(3);
     beta3=Pars(4);
     sigma2u=Pars(5);
     sigma2v=Pars(6);
%     sigma2w=1;%Pars(7);
     
     lambda=sqrt(sigma2u/sigma2v);
     sigma2=sigma2u+sigma2v;
     sigma=sqrt(sigma2);
     
     eps = y-alpha-x1*beta1-x2*beta2-x3*beta3;
     B2 = p2 - p1 +log(beta1)-log(beta2);
     B3 = p3 - p1 +log(beta1)-log(beta3);
     w1 = x1 - x2 - B2;
     w2 = x1 - x3 - B3; 
     w = [w1 , w2];
    
     
     Den = 2/sigma*normpdf(eps/sigma,0,1).*normcdf(-lambda*eps/sigma,0,1);
     logDen = log(Den);

end

