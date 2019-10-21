function tag = getDefaultTag( obj )
% GETDEFAULTTAG  Get default tag of an nlsaLinearMap object
%
% Modified 2014/07/20

tag = idx2str( getBasisFunctionIndices( obj ), 'idxPhi' );

