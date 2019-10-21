function file  = getEigenfunctionFile( obj, iB, iR )
% GETOPERATORFILE  Get eigenfunction filename from an nlsaDiffusionOperator_gl 
% object 
%
% Modified 2014/04/08

if nargin == 2 
     [ iB, iR ] = gl2loc( getPartition( obj ), iB );
end

file = getFile( getEigenfunctionFilelist( obj, iR ), iB );
