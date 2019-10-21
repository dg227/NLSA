function v = torus_v( theta, f, a )
%% VECTOR FIELD ON THE TWO-TORUS
%
% d theta( 1 ) / dt  = 1 + sqrt( 1 - a( 1 ) ) * cos( theta( 1 ) ) 
% d theta( 2 ) / dt  = f * ( 1 - sqrt( 1 - a( 2 ) ) * sin( theta( 2 ) ) )

v = zeros( size( theta ) );
v( 1, : ) = 1 + sqrt( 1 - a( 1 ) ) * cos( theta( 1, : ) );
v( 2, : ) = f * ( 1 - sqrt( 1 - a( 2 ) ) * sin( theta( 2, : ) ) );

