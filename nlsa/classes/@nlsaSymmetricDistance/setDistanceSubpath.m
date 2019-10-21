function obj = setDistanceSubpath( obj, pth )
% SETDISTANCESUBPATH  Set distance subdirectory of nlsaSymmetricDistance object
%
% Modified 2014/04/14

if ~isrowstr( pth )
    error( 'Distance subdirectory must be a character string' )
end
obj.pathYS = pth;
