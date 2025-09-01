function [ d, a1, b1, a2, b2 ] = cldmat_g( idxE, zeta, tol, x1, v1, xi1, x2, v2, xi2 )
%CDMAT_G Lagged cone distance matrix with geometric mean
%
%   Modified 05/14/2014

if nargin == 6
    xi1 = xi1 .^ 2;
    ww  = ldmat( idxE, x1 );
    a1  = bsxfun( @rdivide, ww, xi1' );
    x2  = x1;
elseif nargin == 9
    xi1 = xi1 .^ 2;
    xi2 = xi2 .^ 2;
    ww  = ldmat( idxE, x1, x2 );
    a1  = bsxfun( @rdivide, ww, xi1' );
    a2  = bsxfun( @rdivide, ww, xi2  );
end

nS1 = size( x1, 2 );
nS2 = size( x2, 2 );
nSE1 = nS1 - idxE( end ) + 1;
nSE2 = nS2 - idxE( end ) + 1;

vxL = v1' * x2;
[ I, J ] = ndgrid( 1 : nS1, 1 : nS2 );

vx = zeros( nSE1, nSE2 );
for iE = idxE
    vx = vx + v1( :, iE : iE - 1 + nSE1 )' * x2( :, iE : iE - 1 + nSE2 );
end


if nargin == 6
    vx0 = diag( vx );
elseif nargin == 9
    vx0 = sum( v1 .* x1, 1 );
    vx0 = lsum( vx0, [ idxE( end ) nS1 ], idxE )'; 
end
 
b1    = bsxfun( @minus, vx, vx0 ) .^ 2;
b1    = bsxfun( @rdivide, b1, xi1' ) ./ ww; % cos^2(angle)
ifFix = b1 > 1;
b1( ifFix ) = 1;
b1 = 1 - zeta .* b1; 

if nargin == 9
    vx = zeros( nSE1, nSE2 );
    for iE = idxE
        vx = vx + x1( :, iE : iE - 1 + nSE1 )' * v2( :, iE : iE - 1 + nSE2 );
    end
    vx0         = sum( v2 .* x2, 1 );
    vx0         = lsum( vx0, [ idxE( end ) nS2 ], idxE );
    b2          = bsxfun( @minus, vx, vx0 ) .^ 2;
    b2          = bsxfun( @rdivide, b2, xi2 ) ./ ww;
    ifFix       = b2 > 1;
    b2( ifFix ) = 1;
    b2          = 1 - zeta .* b2; 
end  

if nargin == 6
    d = sqrt( a1 .* a1' .* b1 .* b1' );
elseif nargin == 9
    d = sqrt( a1 .* a2 .* b1 .* b2 );
end

if nargin == 6
    if tol > 0
        if0      = a1 <= tol;
        if0      = if0 | if0';
        d( if0 ) = 0;
    else
        n = size( d, 1 );
        d( 1 : n + 1 : end ) = 0;
    end
elseif nargin == 9 && tol > 0
    if0      = a1 <= tol | a2 <= tol;
    d( if0 ) = 0;
end
