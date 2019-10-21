function res = getFDOrder( obj )
% GEFDORDER  Returns finite-differnce order of an array of nlsaEmbeddedComponent_xi objects
%
% Modified 2013/12/19

res = zeros( size( obj ) );
for iObj = 1 : numel( obj )
    res( iObj ) = obj( iObj ).fdOrd;
end
