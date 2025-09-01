function tag = getDefaultBasisFunctionTag( obj )
% GETDEFAULTBASISFUNCTIONTAG  Get default basis function tag of an 
% nlsaComponent_rec_phi object
%
% Modified 2015/08/27

tag = idx2str( getBasisFunctionIndices( obj ), 'idxPhi' );

