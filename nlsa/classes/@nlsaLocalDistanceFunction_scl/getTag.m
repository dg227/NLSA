function tag = getTag( obj )
% GETTAG  Get tag of an nlsaLocalDistanceFunction_scl object
%
% Modified 2015/10/31

tag = strjoin_e( { getTag@nlsaLocalDistanceFunction( obj ) ...
                   getTag( getLocalScaling( obj ) ) }, '_' );
