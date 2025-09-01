function pth = getDefaultDataSubpath( obj )
% GETDEFAULTDATASUBPATH Get default state subpath of an 
% nlsaEmbeddedComponent_rec object
%
% Modified 2014/08/04

pth = strcat( 'dataX_r_', getDefaultBasisFunctionTag( obj ) );
