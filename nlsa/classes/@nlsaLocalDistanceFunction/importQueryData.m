function obj = importQueryData( obj, dat, iC, iB, iR )
% IMPORTDATA  Retrieve data for distance computation from 
% nlsaEmbeddedComponent batch
%
% Modified 2015/10/29

lDist = getLocalDistance( obj );
obj.Q = importData( lDist, dat, iC, iB, iR );

