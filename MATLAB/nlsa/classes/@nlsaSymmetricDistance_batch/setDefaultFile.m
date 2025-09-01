function obj = setDefaultFile( obj )
% SETDEFAULTFILE Set default filenames for an nlsaSymmetricDistance_batch
% object 
%
% Modified 2014/04/30

obj  = setDistanceFile( obj,  getDefaultDistanceFile( obj ) );
