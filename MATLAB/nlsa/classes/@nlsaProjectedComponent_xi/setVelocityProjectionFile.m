function obj = setVelocityProjectionFile( obj, file )
% SETVELOCITYPROJECTIONFILE  Set velocity projected data file of an 
% nlsaProjectedComponent_xi object
%
% Modified 2014/06/24

if ~isrowstr( file )
    error( 'File property must be a character string' )
end
obj.fileAXi = file;
