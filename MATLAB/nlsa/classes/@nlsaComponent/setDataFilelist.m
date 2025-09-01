function obj = setDataFilelist( obj, file )
% SETDATAFILELIST Set filelist property of an nlsaComponent object
%
% Modified 2014/04/11

if ~isa( file, 'nlsaFilelist' ) || getNFile( file ) ~= getNBatch( obj )
        error( 'Incompatible data filelist' )
end
obj.file = file;
