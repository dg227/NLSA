function [ d, b1, b2 ] = cdmat2_g( zeta, tol, x1, v1, xi1, x2, v2, xi2 )
%CDMAT_G Cone distance matrix with geometric mean and no normalization by the phase velocity magnitude
%
%   Modified 01/30/2016

if zeta == 0; % don't compute angular terms
    if nargin == 5
        d = dmat( x1 );
    elseif nargin == 8
        d = dmat( x1, x2 );
    end
    return
end

if nargin == 5
    xi1 = xi1 .^ 2;
    ww  = dmat( x1 );
    x2  = x1;
elseif nargin == 8
    xi1 = xi1 .^ 2;
    xi2 = xi2 .^ 2;
    ww  = dmat( x1, x2 );
end

vx = v1' * x2;

if nargin == 5
    vx0 = diag( vx );
elseif nargin == 8
    vx0 = sum( v1 .* x1, 1 )';
end
 
b1    = bsxfun( @minus, vx, vx0 ) .^ 2;
b1    = bsxfun( @rdivide, b1, xi1' ) ./ ww; % cos^2(angle)
ifFix = b1 > 1;
b1( ifFix ) = 1;
b1 = 1 - zeta .* b1; 

if nargin == 8
    vx          = x1' * v2;
    vx0         = sum( v2 .* x2 );
    b2          = bsxfun( @minus, vx, vx0 ) .^ 2;
    b2          = bsxfun( @rdivide, b2, xi2 ) ./ ww;
    ifFix       = b2 > 1;
    b2( ifFix ) = 1;
    b2          = 1 - zeta .* b2; 
end  

if nargin == 5
    d = ww .* sqrt( b1 .* b1' );
elseif nargin == 8
    d = ww .* sqrt( b1 .* b2 );
end


if tol > 0
    if0      = ww <= tol;
    d( if0 ) = 0;
end

if tol == 0 && nargin == 5
    n = size( d, 1 );
    d( 1 : n + 1 : end ) = 0;
end
