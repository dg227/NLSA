function q = aggregateProability( p, a, b );
% AGGREGATEPROBABILITY Coarsen the probability vector p for events a based on
% bins b. 

nB = numel( b ) - 1; % number of bins

q = zeros( nB, 1 );

iStart = 1;
for iB = 1 : nB
    iEnd = find( a <= b( iB + 1 ), 1, 'last' );
    q( iB ) = sum( p( iStart : iEnd );
    iStart = iEnd + 1;
end


