function tag = getDefaultEigenfunctionTag( obj )
% GETDEFAULTTEIGENFUNCTIONTAG  Get default eigenfunction tag of an 
% nlsaEmbeddedComponent_ose_n object
%
% Modified 2014/08/04

tag = idx2str( getEigenfunctionIndices( obj ), 'idxPhi' );

