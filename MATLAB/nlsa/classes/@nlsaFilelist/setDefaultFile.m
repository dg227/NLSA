function obj = setDefaultFile( obj, prefix )
% SETDEFAULTFILE Set default filenames for an array of nlsaFilelist objects 
%
% Modified 2014/04/02

if nargin == 1
    prefix = '';
end

file = getDefaultFile( obj, prefix );
obj  = setFile( obj, file );
