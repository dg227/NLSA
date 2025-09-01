function partition = getDenEmbPartitionT( obj )
% GETDENEMBPARTITIONTT Get test partition of the embedded density data of an 
% nlsaModel_den object
%
% Modified 2019/11/24

component = getDenEmbComponentT( obj );
partition = getPartition( component( 1, : ) ); 
