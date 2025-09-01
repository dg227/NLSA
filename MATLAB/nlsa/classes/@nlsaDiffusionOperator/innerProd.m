function c = innerProd( obj, f1, f2 )
% INNERPROD Compute inner product between two sets of 
% scalar functions
% 
% Modified 2014/04/03

mu = getRiemannianMeasure( obj );

f1 = bsxfun( @times, f1, mu );

c = f1' * f2;

