function res = getFDOrder( obj )
% GEFDORDER  Returns finite-differnce order of an array of nlsaKoopmanOperator
% objects.
%
% Modified 2020/04/09

res = zeros( size( obj ) );
for iObj = 1 : numel( obj )
    res( iObj ) = obj( iObj ).fdOrd;
end
