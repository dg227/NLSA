function obj = setPartition( obj, partition )
% SETPARTITION  Set partition of nlsaComponent object
%
% Modified 2020/02/24

if ~isa( partition, 'nlsaPartition' )
    error( 'Second argument must be an array of nlsaPartition objects' )
end

if ~isequal( size( obj ), size( partition ) )
    error( 'Incompatible size of input arguments' )
end

for iObj = 1 : numel( obj )
    if ~isequal( obj( iObj ).partition, partition( iObj ) )
        obj( iObj ).partition = partition( iObj );

        % Reset fileList since partition has changed 
        obj( iObj ).file = nlsaFilelist( ...
            'nFile', getNBatch( partition( iObj ) ) );
    end
end
