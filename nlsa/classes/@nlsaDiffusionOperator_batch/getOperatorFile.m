function file  = getOperatorFile( obj, iB, iR )
% GETOPERATORFILE  Get operator filename from an nlsaDiffusionOperator_batch 
% object 
%
% Modified 2014/04/08

if nargin == 2 
     [ iB, iR ] = gl2loc( getPartition( obj ), iB );
end

file = getFile( getOperatorFilelist( obj, iR ), iB );
