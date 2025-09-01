function path = getPairwiseDistancePath( obj )
% GETPAIRWISEDISTANCEPATH Get path of pairwise distances in nlsaModel object
%
% Modified 2013/04/10

path = getPath( getPairwiseDistance( obj ) ); 
