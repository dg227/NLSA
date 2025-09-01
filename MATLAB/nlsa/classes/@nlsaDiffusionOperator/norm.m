function c = norm( obj, f )
% NORM Compute L2 norm of a set of scalar functions 
% 
% Modified 2013/04/17

c = sqrt( innerProd( obj, f, f ) );

