function tag = getTag( obj )
% GETTAG  Get tag of nlsaLocalDistance_l2_scl object
%
% Modified 2015/10/31

lScl = getLocalScaling (obj );
tag  = [ getTag@nlsaLocalDistance_l2( obj ) '_' getDefaultTag( lScl ) ]; 
