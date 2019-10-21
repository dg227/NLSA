function obj = setRightSingularVectorSubpath( obj, pth )
% SETRIGHTSINGULARVECTORSUBPATH  Set right singular vector subdirectory of an 
% nlsaCovarianceOperator object
%
% Modified 2014/07/16

if ~isrowstr( pth )
    error( 'Path must be a character string' )
end
obj.pathV = pth;
