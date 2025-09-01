function obj = setRightSingularVectorFile( obj, file )
% SETRIGHTSINGULARVECTORFILE  Set files for the right singular vectors of an
% nlsaCovarianceOperator object 
%
% Modified 2014/07/16

if isa( file, 'nlsaFilelist' )
    if ~isCompatible( file, getPartition( obj ) )
        error( 'Incompatible file list' )
    end
    obj.fileV = file;
else
    obj.fileV = setFile( obj.fileV, file );
end

