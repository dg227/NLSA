function p = pc( x, y )
%PC Compute PC score between time series. Time index is taken to be the 
% second index of the arrays.

nX = size( x, 1 );
nS = size( x, 2 );

p = zeros( nX, 1 );
for iX = 1 : nX
    p( iX ) = corr( x( iX, : ).', y( iX, : ).' );
end 

