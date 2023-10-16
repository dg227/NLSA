function dx = rossler(t, x, a, b, c)
% Evaluates the right hand side of the Rossler system
% x' = -y -z
% y' = x + a*y
% z' = b + z*(x - c)
% typical values: a=0.1, b=0.1, c=14

dx = zeros(3, 1);
dx(1) = - x(2) - x(3);
dx(2) = x(1) + a*x(2);
dx(3) = b + x(3)*(x(1) - c);
