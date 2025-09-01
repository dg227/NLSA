function res = getFDType( obj )
% GETFDTYPE  Returns finite-differnce type of an array of 
% nlsaEmbeddedComponent_xi objects
%
% Modified 2013/12/19

if isscalar( obj )
    res = obj.fdType;
    return
end

res = cell( size( obj ) );
for iObj = 1 : numel( obj )
    res{ iObj } = obj( iObj ).fdType;
end
