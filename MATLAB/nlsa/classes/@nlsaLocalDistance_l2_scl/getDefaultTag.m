function tag = getDefaultTag( obj )
% GETDEFAULTTAG  Get default tag of nlsaLocalDistance_l2_scl object
%
% Modified 2015/10/31

lScl = getLocalScaling (obj );
tag  = [ getDefaultTag@nlsaLocalDistance_l2( obj ) '_scl_' getDefaultTag( lScl ) ]; 
