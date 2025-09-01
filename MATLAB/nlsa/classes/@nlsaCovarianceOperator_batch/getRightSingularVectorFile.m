function file  = getRightSingularVectorFile( obj, iB, iR )
% GETRIGHTSINGULARVECTORFILE  Get right singular vector filenames of an 
% nlsaCovarianceOperator object 
%
% Modified 2014/07/16

if nargin == 2 
     [ iB, iR ] = gl2loc( getPartition( obj ), iB );
end

file = getFile( getRightSingularVectorFilelist( obj, iR ), iB );
