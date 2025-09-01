function obj = setLeftSingularVectorSubpath( obj, pth )
% SETLEFTSINGULARVECTORSUBPATH  Set left singular vector subdirectory of an 
% nlsaCovarianceOperator object
%
% Modified 2014/07/16

if ~isrowstr( pth )
    error( 'Path must be a character string' )
end
obj.pathU = pth;
