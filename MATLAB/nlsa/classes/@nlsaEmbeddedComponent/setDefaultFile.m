function obj = setDefaultFile( obj )
% SETDEFAULTFILE Set default filenames for an nlsaEmbeddedComponent object 
%
% Modified 2014/03/30

obj = setDefaultFile@nlsaComponent( obj );
obj = setDataFile_before( obj, getDefaultDataFile_before( obj ) );
obj = setDataFile_after( obj, getDefaultDataFile_after( obj ) );
