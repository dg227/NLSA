function partition = getEmbPartition( obj )
% GETEMBPARTITION Get partition of the embedded data of an nlsaModel_base object
%
% Modified 2014/05/21

comp = getEmbComponent( obj );
partition = getPartition( comp( 1, : ) ); 
