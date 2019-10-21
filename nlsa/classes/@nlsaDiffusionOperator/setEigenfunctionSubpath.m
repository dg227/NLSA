function obj = setOperatorSubpath( obj, pth )
% SETEIGENFUNCTIONSUBPATH  Set eigenfunction subdirectory of nlsaDiffusionOperator object
%
% Modified 2014/04/03

if ~isrowstr( pth )
    error( 'Path must be a character string' )
end
obj.pathPhi = pth;
