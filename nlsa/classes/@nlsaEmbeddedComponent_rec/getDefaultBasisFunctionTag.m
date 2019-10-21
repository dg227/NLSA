function tag = getDefaultBasisFunctionTag( obj )
% GETDEFAULTTAG  Get default basis function tag of an 
% nlsaEmbeddedComponent_rec object
%
% Modified 2014/08/04

tag = idx2str( getBasisFunctionIndices( obj ), 'idxPhi' );

