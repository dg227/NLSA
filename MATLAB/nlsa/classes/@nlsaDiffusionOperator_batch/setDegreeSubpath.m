function obj = setDegreeSubpath( obj, pth )
% SETDEGREESUBPATH  Set degree subdirectory of an 
% nlsaDiffusionOperator_batch object
%
% Modified 2014/04/09

if ~isrowstr( pth )
    error( 'Path must be a character string' )
end
obj.pathD = pth;
