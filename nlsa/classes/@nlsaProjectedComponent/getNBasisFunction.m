function nL = getNBasisFunction( obj )
% GETNBASISFUNCTION Returns the number of basis functions of an array of 
% nlsaProjectedComponent objects
%
% Modified 2016/04/05

nL = zeros( size( obj ) );
for iObj = 1 : numel( obj )
    nL( iObj ) = obj( iObj ).nL;
end
