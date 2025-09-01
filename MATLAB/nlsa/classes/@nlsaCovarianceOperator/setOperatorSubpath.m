function obj = setOperatorSubpath( obj, pth )
% SETOPERATORSUBPATH  Set operator subdirectory of an nlsaCovarianceOperator 
% object
%
% Modified 2014/07/16

if ~isrowstr( pth )
    error( 'Path must be a character string' )
end
obj.pathA = pth;
