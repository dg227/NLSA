function nD = getDimension( obj )
% GETDIMENSION  Get dimension of nlsaKernelDensity objects
%
% Modified 2015/04/06

nD = zeros( size( obj ) );
for iObj = 1 : numel( obj )
    nD( iObj ) = obj( iObj ).nD;
end
