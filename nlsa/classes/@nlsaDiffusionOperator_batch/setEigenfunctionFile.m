function obj = setEigenfunctionFile( obj, file )
% SETEIGENFUNCTIONFILE  Set files for the diffusion eigenfunctions in an
% nlsaDiffusionOperator_batch object 
%
% Modified 2014/06/13

if isa( file, 'nlsaFilelist' )
    if ~isCompatible( file, getPartition( obj ) )
        error( 'Incompatible file list' )
    end
    obj.filePhi = file;
else
    obj.filePhi = setFile( obj.filePhi, file );
end

