function setDistances( obj, yVal, yInd, iB, varargin )
% SETDISTANCES  Set batch data of an nlsaPairwiseDistance object 
%
% Modified 2014/04/15

sizYVal = size( yVal );
sizYInd = size( yInd );

if   numel( sizYVal ) ~= numel( sizYInd ) ...
  || any( sizYVal ~= sizYInd )
    error( 'Incompatible distance and nearest neighbor index data' )
end

if ~ischar( varargin{ 1 } )
    iB = { iB, varargin{ 1 } }; % varargin{ 1 } stores realization
    varargin = varargin( 2 : end );
else
    iB = { iB };
end

[ sizB( 1 ), sizB( 2 ) ] = getBatchArraySize( obj, iB{ : } ); 
if any( sizYVal ~= sizB )
    error( 'Incompatible size of distance data array' )
end
    
file = fullfile( getDistancePath( obj ), ... 
                 getDistanceFile( obj, iB{ : } ) );

save( file, 'yVal', 'yInd', varargin{ : } )

