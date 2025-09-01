function obj = setPartition( obj, partition )
% SETPARTITION  Set partition of nlsaEmbeddedComponent_xi object
%
% Modified 2014/04/11

obj = setPartition@nlsaEmbeddedComponent( obj, partition );
if getNBatch( partition ) ~= getNFile( getVelocityFilelist( obj ) )
    obj = setVelocityFile( obj, nlsaFilelist( 'nFile', getNBatch( partition ) ) );
end
