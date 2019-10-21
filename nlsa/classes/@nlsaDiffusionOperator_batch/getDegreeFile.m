function file  = getDegreeFile( obj, iB, iR )
% GETDEGREEFILE  Get degree filename from an nlsaDiffusionOperator_batch
% object 
%
% Modified 2014/04/08

if nargin == 2 
     [ iB, iR ] = gl2loc( getPartition( obj ), iB );
end

file = getFile( getDegreeFilelist( obj, iR ), iB );
