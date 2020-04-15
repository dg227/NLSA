function obj = setEigenfunctionSubpath( obj, pth )
% SETEIGENFUNCTIONSUBPATH  Set eigenfunction subdirectory of 
% nlsaKoopmanOperator object
%
% Modified 2020/04/15

if ~isrowstr( pth )
    error( 'Path must be a character string' )
end
obj.pathEig = pth;
