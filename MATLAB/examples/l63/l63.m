function dx = l63( t, x, sigma, rho, beta )
% Evaluates the right hand side of the Lorenz system
% x' = sigma*(y-x)
% y' = x*(rho - z) - y
% z' = x*y - beta*z
% typical values: rho = 28; sigma = 10; beta = 8/3;

dx = zeros( 3, 1 );
dx( 1 ) = sigma * ( x( 2 ) - x( 1 ) );
dx( 2 ) = x( 1 ) * ( rho - x(3) ) - x( 2 );
dx( 3 ) = x( 1 ) * x( 2 ) - beta * x( 3 );
