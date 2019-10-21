function theta = torus_orbit( t, f, a );
%% DYNAMICAL SYSTEM ON THE TWO-TORUS
%
% d theta( 1 ) / dt  = 1 + sqrt( 1 - a( 1 ) ) * cos( theta( 1 ) ) 
% d theta( 2 ) / dt  = f * ( 1 - sqrt( 1 - a( 2 ) ) * sin( theta( 2 ) ) )
%
% Initial conditions:
% theta( 1 ) = 0
% theta( 2 ) = 0
%
% The parameter values a = [ 1 1 ] correspond to linear flow on the torus

theta = zeros( 2, numel( t ) );
if a( 1 ) ~= 1
    theta( 1, : ) = 2 * atan( ( 1 + sqrt( 1 - a( 1 ) ) ) ...
                              * tan( sqrt( a( 1 ) ) * t / 2 ) ...
                              / sqrt( a( 1 ) ) );
else
    theta( 1, : ) = t;
end
if a( 2 ) ~= 1
    theta( 2, : ) = 2 * acot( ( sqrt( 1 - a( 2 ) ) ) + sqrt( a( 2 ) ) ...
                              * cot( sqrt( a( 2 ) ) * f * t / 2 ) ); 
else
    theta( 2, : ) = f * t + pi / 2;
end
