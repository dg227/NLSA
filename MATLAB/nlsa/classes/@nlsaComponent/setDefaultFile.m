function obj = setDefaultFile( obj )
% SETDEFAULTFILE Set default filenames for an nlsaComponent object 
%
% Modified 2014/04/05

obj = setDataFile( obj, getDefaultDataFile( obj ) );
