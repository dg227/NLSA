function obj = setVelocityFile( obj, file )
% SETVELOCITYFILE Set phase space velocity filenames of an 
% nlsaEmbeddedComponent_xi object
%
% Modified 2014/04/06

if isa( file, 'nlsaFilelist' )
    if getNFile( file ) ~= getNBatch( obj )
        error( 'Incompatible number of files' )
    end
    obj.fileXi = file;
elseif isrowstr( file ) || iscellstr( file )
    obj.fileXi = setFile( obj.fileXi, file );
else
    error( 'Invalid filename specification' )
end

