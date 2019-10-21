function tag = getDefaultTag( obj )
% GETDEFAULTTAG  Get default tag of an nlsaLocalDistanceFunction_scl object
%
% Modified 2015/10/31

tag = strjoin_e( { getDefaultTag@nlsaLocalDistanceFunction( obj ) ...
                   getDefaultTag( getLocalScaling( obj ) ) }, '_' );
