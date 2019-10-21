function partition = getOutEmbPartition( obj )
% GETOSEEMBPARTITION Get partition of the embedded OS data of an 
% nlsaModel_ose object
%
% Modified 2014/05/21

comp = getOutEmbComponent( obj );
partition = getPartition( comp( 1, : ) ); 
