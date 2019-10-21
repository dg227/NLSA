function nC = getNComponent( obj )
% GETNCOMPONENT  Get number of components in an nlsaLocalDistanceData object
%
% Modified  2015/10/23

nC = getNComponent( getComponent( obj ) ); 
