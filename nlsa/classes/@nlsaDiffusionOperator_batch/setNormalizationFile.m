function obj = setNormalizationFile( obj, file )
% SETNORMALIZATIONFILE  Set files for the kernel normalization in an
% nlsaDiffusionOperator_batch object 
%
% Modified 2018/06/18

if isa( file, 'nlsaFilelist' )
    if ~isCompatible( file, getPartition( obj ) )
        error( 'Incompatible file list' )
    end
    obj.fileQ = file;
else
    obj.fileQ = setFile( obj.fileQ, file );
end
