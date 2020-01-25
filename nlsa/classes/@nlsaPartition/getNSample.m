function nS = getNSample( obj, iR )
% GETNSAMPLE  Get number of samples in an array of nlsaPartition objects
%
% Modified  2020/01/21
 
%% Two input arguments (explicit elements within object array specified)
if nargin == 2
    nS = getNSample( obj( iR ) );
    return
end

%% One input argument (number of samples for all elements in array returned) 

% Return 0 samples if the object is empty
if isempty( obj )
    nS = 0;
    return
end

% Compute number of samples for each object in the array
nS = zeros( size( obj ) );
for iObj = 1 : numel( obj )
    nS( iObj ) = obj( iObj ).idx( end );
end
