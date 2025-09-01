function xNorm = norm( obj, x )
% NORM Norm of embedded data in explicit storage format
% 
% Modified 2014/04/05

xNorm = sqrt( norm2( obj, x ) );
