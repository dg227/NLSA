function obj = setStateErrorFile( obj, file )
% SETSTATEERRORFILE Set state error filenames of an nlsaEmbeddedComponent_ose 
% object
%
% Modified 2014/05/20

if isa( file, 'nlsaFilelist' )
    if getNFile( file ) ~= getNBatch( obj )
        error( 'Incompatible number of files' )
    end
    obj.fileEX = file;
elseif isrowstr( file ) || iscellstr( file )
    obj.fileEX = setFile( obj.fileEX, file );
else
    error( 'Invalid filename specification' )
end

