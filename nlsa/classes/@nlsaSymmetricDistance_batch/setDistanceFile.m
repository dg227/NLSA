function obj = setDistanceFile( obj, file )
% SETDISTANCEFILE  Set files for pairwise distances in an  
% nlsaSymmetricDistance_batch object 
%
% Modified 2014/06/013

if isa( file, 'nlsaFilelist' )
    if ~isCompatible( file, getPartition( obj ) )
        error( 'Incompatible file list' )
    end
    obj.file = file;
else
    obj.file = setFile( obj.file, file );
end

