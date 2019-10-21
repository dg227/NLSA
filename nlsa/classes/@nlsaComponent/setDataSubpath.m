function obj = setDataSubpath( obj, pthX )
% SETDATASUBPATH  Set pathX of nlsaComponent object
%
% Modified 2013/12/21

if ~ischar( pthX )
    error( 'pathX property must be a character string' )
end
obj.pathX = pthX;
