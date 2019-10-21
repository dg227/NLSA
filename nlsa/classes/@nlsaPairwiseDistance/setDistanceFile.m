function obj = setDistanceFile( obj, file )
% SETDISTANCEFILE  Set files for pairwise distances in an nlsaPairwiseDistance 
% object 
%
% Modified 2014/06/13

if isa( file, 'nlsaFilelist' )
    if ~isCompatible( file, getPartition( obj ) )
        error( 'Incompatible file list' )
    end
    obj.file = file;
else
    obj.file = setFile( obj.file, file );
end
   

