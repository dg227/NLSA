function f = unitaryForecast( c, phi, omega, t )
% UNITARYFORECAST Compute the evolution at times t  of an observable with 
% expansion coefficients c in an orthonormal basis phi, under unitary 
% dynamics with eigenfrequencies omega
%
% Input arguments
%
% c:     [ nL nD ] sized array containing the expansion coefficients of the 
%        observable, where nD is the spatial dimenaion and nL the number of 
%        basis functions.
%
% phi:   [ nS nL ] sized array containing the values of the basis functions,
%        where nS is the number of initial conditions.
%
% omega: [ nL ] sized vector containing the eigenfrequencies.
%
% t:     [ nT ] sized vector containing the forecast lead times.
%
% Output arguments
%
% f:     [ nS nD nT ] sized array containing the forecast values.
%
% Modified 2020/07/24

[ nL, nD ] = size( c );
nT = numel( t );

omega = reshape( omega, [ nL 1 ] );
t = reshape( t, [ 1 1 nT ] ); 

cT = c .* exp( i * omega .* t );
cT = reshape( cT, [ nL, nD * nT ] );

f = phi * cT;
f = reshape( f, [ nS nD nT ] );



