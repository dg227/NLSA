function D = importData( obj, dat, iC, iB, iR )
%% IMPORTDATA  Get distance data from nlsaEmbeddedComponent_xi batch
%
% Modified 2015/10/23
            
comp = getComponent( dat );
D = importData@nlsaLocalDistance_l2( obj, dat, iC, iB, iR );
D.xiNorm = getVelocityNorm( comp( iC, iR ), iB ); 
