function obj = setLeftSingularVectorFile( obj, file )
% SETLEFTSINGULARVECTORFILE  Set files for the left singular vectors of an
% nlsaCovarianceOperator object 
%
% Modified 2014/07/16


if isa( file, 'nlsaFilelist' )
    if ~isCompatible( file, getPartition( obj ) )
        error( 'Incompatible file list' )
    end
    obj.fileU = file;
else
    obj.fileU = setFile( obj.fileU, file );
end

