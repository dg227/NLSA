function D = importData( obj, dat, iC, iB, iR )
% IMPORTDATA  Retrieve data for distance computation from 
% nlsaLocalDistanceData_scl batch
%
% Modified 2015/10/31

lScl = getLocalScaling( obj );
D   = importData@nlsaLocalDistance_l2( obj, dat, iC, iB, iR );
D.S = importData( lScl, dat, iC, iB, iR );

