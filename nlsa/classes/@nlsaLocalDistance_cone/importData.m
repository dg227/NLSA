function D = importData( obj, dat, iC, iB, iR )
%% IMPORTDATA  Get velocity data from nlsaEmbeddedComponent_xi batch
%
% Modified 2017/07/21

D = importData@nlsaLocalDistance_at( obj, dat, iC, iB, iR );

switch getMode( obj )
    case 'explicit'
        outFormat = 'evector';
    case 'implicit'
        outFormat = 'overlap';
        
end

if getZeta( obj ) > 0
    comp = getComponent( dat );
    [ ~, D.xi ] = getVelocity( comp( iC, iR ), iB, outFormat );
else
    D.xi = [];
end
