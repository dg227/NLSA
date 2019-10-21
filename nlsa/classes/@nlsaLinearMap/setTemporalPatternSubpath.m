function obj = setTemporalPatternSubpath( obj, pth )
% SETTEMPORALPATTERNSUBPATH  Set temporal pattern subdirectory of an 
% nlsaLinearMap object
%
% Modified 2015/10/19

if ~isrowstr( pth )
    error( 'Path must be a character string' )
end
obj.pathVT = pth;
