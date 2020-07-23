function n = getNBasisFunction( obj )
% GETNBASISFUNCTION Returns the number of basis functions of an
% nlsaKoopmanOperator object
%
% Modified 2020/07/22

n = numel( getBasisFunctionIndices( obj ) );
