function nD = getDimension( obj )
% GETDIMENSION  Get dimension of nlsaComponent object
%
% Modified 2012/12/20

nD = zeros( size( obj ) );
for iObj = 1 : numel( obj )
    nD( iObj ) = obj( iObj ).nD;
end
