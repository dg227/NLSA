function partition = getOutDenEmbPartition( obj )
% GETOUTDENEMBPARTITION Get partition of the embedded OS data for density
% estimation of an nlsaModel_den_ose object
%
% Modified 2020/01/25

comp = getOutDenEmbComponent( obj );
partition = getPartition( comp( 1, : ) ); 
