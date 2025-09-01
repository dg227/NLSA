function obj = setDefaultSubpath( obj )
% SETDEFAULTSUBPATH Set default subpath of nlsaSymmetricDistance object
%
% Modified 2014/04/03

obj = setDistanceSubpath( obj, getDefaultDistanceSubpath( obj ) );

