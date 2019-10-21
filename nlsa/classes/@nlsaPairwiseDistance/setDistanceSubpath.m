function obj = setDistanceSubpath( obj, pthY )
% SETDISTANCESUBPATH  Set distance subdirectory of nlsaPairwiseDistance object
%
% Modified 2014/02/10

if ~ischar( pthY )
    error( 'Distance subdirectory must be a character string' )
end
obj.pathY = pthY;
