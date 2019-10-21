function D = importData( obj, dat, iC, iB, iR )
% IMPORTDATA  Retrieve data for distance computation from 
% nlsaEmbeddedComponent batch
%
% Modified 2015/10/31

comp = getComponent( dat );
switch getMode( obj )
    case 'explicit'
        outFormat = 'evector';
    case 'implicit'
        outFormat = 'overlap';
end

D.x    = getData( comp( iC, iR ), iB, outFormat );
D.idxE = getEmbeddingIndices( comp( iC, iR ) );
