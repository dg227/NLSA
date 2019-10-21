function obj = setDefaultTag( obj )
% SETDEFAULTTAG  Set default tag of nlsaLocalDistance_l2_scl object
%
% Modified 2015/10/30

obj = setDefaultTag@nlsaLocalDistance_l2( obj );
obj.lScl = setDefaultTag( obj.lScl );

