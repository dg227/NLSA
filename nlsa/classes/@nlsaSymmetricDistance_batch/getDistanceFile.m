function file = getDistanceFile( obj, iB, iR )
% GETDISTANCEFILE  Get distance filename from an nlsaSymmetricDistance_batch
% object 
%
% Modified 2014/04/30

if nargin == 2 
     [ iB, iR ] = gl2loc( getPartition( obj ), iB );
end

file = getFile( getDistanceFilelist( obj, iR ), iB );
