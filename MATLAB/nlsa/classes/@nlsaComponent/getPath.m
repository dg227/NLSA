function path = getPath( obj, idxC, idxR )
% GETPATH Get path property of an array of nlsaComponent objects
%
% Modified 2013/10/14

[ nC, nR ] = size( obj );


if nargin < 3
    idxR = 1 : nR;
end
if nargin < 2
    idxC = 1 : nC;
end
        
nCOut = numel( idxC );
nROut = numel( idxR );

path = cell( nCOut, nROut );
for iR = 1 : nROut
    for iC = 1 : nCOut
        path{ iC, iR } = obj( iC, iR ).path;
    end
end
            
if isscalar( path )
    path = path{ 1 };
end
