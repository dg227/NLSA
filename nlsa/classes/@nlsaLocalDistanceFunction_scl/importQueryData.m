function obj = importQueryData( obj, dat, iC, iB, iR )
% IMPORTQUERYDATA  Retrieve query data for distance computation  
%
% Modified 2015/10/31

obj    = importQueryData@nlsaLocalDistanceFunction( obj, dat, iC, iB, iR );
lScl   = getLocalScaling( obj );
obj.QS = importData( lScl, dat, iC, iB, iR );

