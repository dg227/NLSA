function obj = setOperatorFile( obj, file )
% SETOPERATORFILE  Set files for the heat kernel in an
% nlsaDiffusionOperator_batch object 
%
% Modified 2014/06/13

if isa( file, 'nlsaFilelist' )
    if ~isCompatible( file, getPartition( obj ) )
        error( 'Incompatible file list' )
    end
    obj.fileP = file;
else
    obj.fileP = setFile( obj.fileP, file );
end