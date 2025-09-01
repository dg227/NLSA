function D = importData( obj, comp, iC, iB, iR )
%% IMPORTDATA  Get scaling data from nlsaEmbeddedComponent_ose batch
%
% Modified 2014/06/18

[ bX, bXi ] = getConstant( obj );

D.xiRef = getReferenceVelocityNorm( comp( iC, iR ), iB );

if bX ~= 0
    D.x = getStateErrorNorm( comp( iC, iR ), iB ); 
end

if bXi ~= 0
    D.xi = getVelocityErrorNorm( comp( iC, iR ), iB );
end

