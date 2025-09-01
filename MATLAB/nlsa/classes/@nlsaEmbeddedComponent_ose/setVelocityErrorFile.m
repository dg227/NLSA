function obj = setVelocityErrorFile( obj, file )
% SETVELOCITYERRORFILE Set phase space velocity error filenames of an 
% nlsaEmbeddedComponent_ose object
%
% Modified 2014/04/06

if isa( file, 'nlsaFilelist' )
    if getNFile( file ) ~= getNBatch( obj )
        error( 'Incompatible number of files' )
    end
    obj.fileEXi = file;
elseif isrowstr( file ) || iscellstr( file )
    obj.fileEXi = setFile( obj.fileEXi, file );
else
    error( 'Invalid filename specification' )
end

