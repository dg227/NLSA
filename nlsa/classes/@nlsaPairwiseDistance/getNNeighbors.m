function nN = getNNeighbors( obj )
% GETNNEIGHBORS  Get number of nearest neighrbors in an array of 
% nlsaPairwiseDistance objects
%
% Modified 2019/11/05

nN = zeros( size( obj ) );
for iObj = 1 : numel( obj )
    nN( iObj ) = obj( iObj ).nN;
end
