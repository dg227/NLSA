function tag = getDefaultOseTag( obj )
% GETDEFAULTOSETAG Get default OSE tag of an nlsaEmbeddedComponent_ose_n object
%
% Modified 2015/12/14

tag = strcat( 'nystrom_', idx2str( getEigenfunctionIndices( obj ), 'idxPhi' ) );
