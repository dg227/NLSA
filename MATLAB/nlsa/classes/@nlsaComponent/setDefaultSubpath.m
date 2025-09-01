function obj = setDefaultSubpath( obj )
% SETDEFAULTSUBPATH Set default subpath of nlsaComponent object
%
% Modified 2015/09/09

obj = setDataSubpath( obj, getDefaultDataSubpath( obj ) );

