function [ d, a1, b1, a2, b2 ] = cldmat_g( idxE, zeta, tol, x1, v1, xi1, x2, v2, xi2 )
%CDMAT_G Lagged cone distance matrix with geometric mean and no normalization by the phase velocity magnitude
%
%   Modified 2018/05/04

if zeta == 0; % don't compute angular terms
    if nargin == 6
        d  = ldmat( idxE, x1 );
    elseif nargin == 9
        d  = ldmat( idxE, x1, x2 );
    end
    return
end

if nargin == 6
    xi1 = xi1 .^ 2;
    ww  = ldmat( idxE, x1 );
    x2  = x1;
elseif nargin == 9
    xi1 = xi1 .^ 2;
    xi2 = xi2 .^ 2;
    ww  = ldmat( idxE, x1, x2 );
end

nS1 = size( x1, 2 );
nS2 = size( x2, 2 );
nSE1 = nS1 - idxE( end ) + 1;
nSE2 = nS2 - idxE( end ) + 1;

vx = lsum2( v1' * x2, idxE );

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
    vx          = lsum2( x1' * v2, idxE );
    vx0         = sum( v2 .* x2, 1 );
    vx0         = lsum( vx0, [ idxE( end ) nS2 ], idxE );
    b2          = bsxfun( @minus, vx, vx0 ) .^ 2;
    b2          = bsxfun( @rdivide, b2, xi2 ) ./ ww;
    ifFix       = b2 > 1;
    b2( ifFix ) = 1;
    b2          = 1 - zeta .* b2; 
end  

if nargin == 6
    d = ww .* sqrt( b1 .* b1' );
elseif nargin == 9
    d = ww .* sqrt( b1 .* b2 );
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
