function obj = setOperatorSubpath( obj, pth )
% SETOPERATORSUBPATH  Set operator subdirectory of nlsaDiffusionOperator object
%
% Modified 2014/04/03

if ~isrowstr( pth )
    error( 'Path must be a character string' )
end
obj.pathP = pth;
