function file  = getLeftSingularVectorFile( obj, iB )
% GETLEFTSINGULARVECTORFILE  Get left singular vector filenames of an 
% nlsaCovarianceOperator object 
%
% Modified 2014/07/16

file = getFile( getLeftSingularVectorFilelist( obj ), iB );
