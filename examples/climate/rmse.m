function err = rmse( x, y )
%RMSE Compute RMSE between time series. Time is taken to be the second 
% second dimension of the arrays.

nS  = size( x, 2 );
err = vecnorm( x - y, 2, 2 ) / sqrt( nS );
