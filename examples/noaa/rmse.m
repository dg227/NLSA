function err = rmse( x, y )
%RMSE Compute RMSE between time series. Time index is taken to be the 
% second index of the arrays.

nS  = size( x, 2 );
err = vecnorm( x - y, 2, 2 ) / sqrt( nS );
