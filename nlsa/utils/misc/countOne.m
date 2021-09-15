function w = countOne( w, n )
% Returns the next word of size equal to numel( w ) in an the alphabet of 
% integers -n, -n + 1, ...., n - 1, n 

if w( end ) < n
    w( end ) = w( end ) + 1;
else
    w( end ) = -n;
    w( 1 : end - 1 ) = countOne( w( 1 : end - 1 ), n );
end
