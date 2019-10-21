function obj = importTestData( obj, dat, iC, iB, iR )
% IMPORTTESTDATA  Retrieve data for distance computation from 
% nlsaEmbeddedComponent batch
%
% Modified 2015/10/29

lDist = getLocalDistance( obj );
obj.T = importData( lDist, dat, iC, iB, iR );

