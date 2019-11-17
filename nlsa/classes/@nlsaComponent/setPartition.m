function obj = setPartition( obj, partition )
% SETPARTITION  Set partition of nlsaComponent object
%
% Modified 2019/11/16

if ~isa( partition, 'nlsaPartition' )
    error( 'Second argument must be an array of nlsaPartition objects' )
end

if ~isequal( size( obj ), size( partition ) )
    error( 'Incompatible size of input arguments' )
end

for iObj = 1 : numel( obj )
    obj( iObj ).partition = partition( iObj );
end
