function obj = setDefaultSubpath( obj )
% SETDEFAULTSUBPATH Set default subpath of nlsaPairwiseDistance object
%
% Modified 2014/02/10

obj = setDistanceSubpath( obj, getDefaultDistanceSubpath( obj ) );

