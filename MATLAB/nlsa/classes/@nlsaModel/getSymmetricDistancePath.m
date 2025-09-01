function path = getSymmetricDistancePath( obj )
% GETSYMMETRICDISTANCEPATH Get path of symmetrized distances in nlsaModel object
%
% Modified 2014/04/10

path = getPath( getSymmetricDistance( obj ) );
