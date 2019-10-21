function rho = torus_rho( theta, f, a )
%% EQUILIBRIUM MEASURE RELATIVE TO UNIT-VOLUME FLAT METRIC

% d theta( 1 ) / dt  = 1 + sqrt( 1 - a( 1 ) ) * cos( theta( 1 ) ) 
% d theta( 2 ) / dt  = f * ( 1 - sqrt( 1 - a( 2 ) ) * sin( theta( 2 ) ) )

C = sqrt( ( 1 - ( 1 - a( 1 ) ) ^ 2 ) * ( 1 - ( 1 - a( 2 ) ^ 2 ) ) ); 
rho = C ./ ( ( 1 + sqrt( 1 - a( 1 ) ) * cos( theta( 1, : ) ) ) ...
       .*    ( 1 - sqrt( 1 - a( 2 ) ) * sin( theta( 2, : ) ) ) );

