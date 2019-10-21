function obj = setSingularValueSubpath( obj, pth )
% SETSINGULARVALUESUBPATH  Set singular value subdirectory of an 
% nlsaCovarianceOperator object
%
% Modified 2014/07/16

if ~isrowstr( pth )
    error( 'Path must be a character string' )
end
obj.pathS = pth;
