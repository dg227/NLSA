function obj = setOperatorSubpath( obj, pth )
% SETOPERATORSUBPATH  Set operator subdirectory of nlsaKoopmanOperator object
%
% Modified 2020/04/15

if ~isrowstr( pth )
    error( 'Path must be a character string' )
end
obj.pathOp = pth;
