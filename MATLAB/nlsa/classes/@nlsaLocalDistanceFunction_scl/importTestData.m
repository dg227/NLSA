function obj = importTestData( obj, dat, iC, iB, iR )
% IMPORTTESTDATA  Retrieve test data for distance computation  
%
% Modified 2015/10/31

obj    = importTestData@nlsaLocalDistanceFunction( obj, dat, iC, iB, iR );
lScl   = getLocalScaling( obj );
obj.QT = importData( lScl, dat, iC, iB, iR );

