function D = importData( obj, comp, iC, iB, iR )
%% IMPORTDATA  Get velocity data from nlsaEmbeddedComponent_xi batch
%
% Modified 2015/01/06
            
D = importData@nlsaLocalDistance_at( obj, comp, iC, iB, iR );

switch getMode( obj )
    case 'explicit'
        outFormat = 'evector';
    case 'implicit'
        outFormat = 'overlap';
        
end
[ ~, D.xi ] = getVelocity( comp( iC, iR ), iB, outFormat );
