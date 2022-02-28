function dx = skewRot2( t, x, a1, a2, a3, b1, b2 )
% Skew-rotation on 3-torus.
%
% The base dynamics is a rotation on the 2-torus.
dx = zeros( 3, 1 );
dx( 1 ) = a1;
dx( 2 ) = a2;
dx( 3 ) = a3 + b1 * cos( x( 1 ) ) + b2 * cos( x( 2 ) ); 

