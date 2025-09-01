function [ c, res, resNorm ] = innerProdEigenfunc( obj, f )
% INNERPRODEIGENFUNC Compute inner product between eigenfunctions and a
% collection of functions f 
% 
% Modified 2014/04/03

[ phi, mu ] = getEigenfunctions( obj );

phiMu = bsxfun( @times, phi, mu );

c       = f' * phiMu;
res     = f - phi * c';
resNorm = sum( bsxfun( @times, res, mu ), 1 ); 
