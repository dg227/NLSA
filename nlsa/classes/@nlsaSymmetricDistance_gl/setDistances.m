function setDistances( obj, yRow, yCol, yVal, varargin )
% SETDISTANCES Write distance data of an nlsaSymmetricDistance_gl object
%
% Modified 2014/04/17

numelMin = getNTotalSample( obj ) * getNNeighbors( obj );

if ~iscolumn( yVal ) ...
  || ~isreal( yVal ) ...
  || numel( yVal ) < numelMin 
    error( 'Distance values must be set to a column vector of real numbers of size at least nN * nS' )
end

if any( size( yVal ) ~= size( yRow ) ) ...
  || any( size( yVal ) ~= size( yCol ) )
    error( 'Incompatible sizes of distance values and neighbor indices' )
end


save( fullfile( getDistancePath( obj ), getDistanceFile( obj ) ), ...
               'yRow', 'yCol', 'yVal', varargin{ : } )
