function path = getOutSymmetricDistancePath( obj )
% GETOUTSYMMETRICDISTANCEPATH Get path of symmetrized distances for the 
% model (OS) data in nlsaModel_err object
%
% Modified 20141/05/25

path = getPath( getOutSymmetricDistance( obj ) );
