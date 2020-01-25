function partition = getEmbPartitionT( obj )
% GETEMBPARTITION Get test partition of the embedded data of an nlsaModel_base
% object
%
% Modified 2020/01/25

component = getEmbComponentT( obj );

if isempty( component )
    partition = nlsaPartition.empty;
    return
end

partition = getPartition( component( 1, : ) ); 
