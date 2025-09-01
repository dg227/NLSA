function nR = getNRealization( obj )
% GETNREALIZATION  Get number of realizations in an nlsaLocalDistanceData object
%
% Modified  2015/10/23

nR = getNRealization( getComponent( obj ) ); 
