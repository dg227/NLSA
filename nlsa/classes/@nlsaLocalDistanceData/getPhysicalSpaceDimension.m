function nD = getPhysicalSpaceDimension( obj )
% GETPHYSICALSPACEDIMENSION  Get physical space dimension of 
% nlsaLocalDistanceData object
%
% Modified  2015/10/23

comp = getComponent( obj );
nD = getPhysicalSpaceDimension( comp( :, 1 ) ); 
