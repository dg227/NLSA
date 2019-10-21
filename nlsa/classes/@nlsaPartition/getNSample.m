function nS = getNSample( obj, iR )
% GETNSAMPLE  Get number of samples in an array of nlsaPartition objects
%
% Modified  2017/07/21
 
if nargin == 2
    nS = getNSample( obj( iR ) );
end

nS = zeros( size( obj ) );
for iObj = 1 : numel( obj )
    nS( iObj ) = obj( iObj ).idx( end );
end
