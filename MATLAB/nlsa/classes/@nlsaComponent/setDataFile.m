function obj = setDataFile( obj, file )
% SETDATAFILE Set filenames of nlsaComponent object
%
% Modified 2014/04/06

if isa( file, 'nlsaFilelist' )
    if getNFile( file ) ~= getNBatch( obj )
        error( 'Incompatible number of files' )
    end
    obj.file = file;
elseif isrowstr( file ) || iscellstr( file )
    obj.file = setFile( obj.file, file );
else
    error( 'Invalid filename specification' )
end

