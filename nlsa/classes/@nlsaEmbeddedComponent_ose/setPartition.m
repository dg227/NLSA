function obj = setPartition( obj, partition )
% SETPARTITION  Set partition of nlsaEmbeddedComponent_ose object
%
% Modified 2016/01/25


obj = setPartition@nlsaEmbeddedComponent_xi_e( obj, partition );
if getNBatch( partition ) ~= getNFile( getVelocityErrorFilelist( obj ) )
    obj = setStateErrorFile( obj, nlsaFilelist( 'nFile', getNBatch( partition ) ) );
    obj = setVelocityErrorFile( obj, nlsaFilelist( 'nFile', getNBatch( partition ) ) );
end
