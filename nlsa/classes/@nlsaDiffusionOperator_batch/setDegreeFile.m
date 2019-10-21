function obj = setDegreeFile( obj, file )
% SETDEGREEFILE  Set files for the kernel degree in an
% nlsaDiffusionOperator_batch object 
%
% Modified 2016/01/25

if isa( file, 'nlsaFilelist' )
    if ~isCompatible( file, getPartition( obj ) )
        error( 'Incompatible file list' )
    end
    obj.fileD = file;
else
    obj.fileD = setFile( obj.fileD, file );
end
