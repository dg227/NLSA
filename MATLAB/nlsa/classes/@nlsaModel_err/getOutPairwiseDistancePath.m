function path = getOutPairwiseDistancePath( obj )
% GETOUTPAIRWISEDISTANCEPATH Get path of pairwise distances for the OS data 
% in an nlsaModel_err object
%
% Modified 2014/05/25

path = getPath( getOutPairwiseDistance( obj ) ); 
