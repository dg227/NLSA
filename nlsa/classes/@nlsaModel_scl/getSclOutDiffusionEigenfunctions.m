function [ phi, mu, lambda ] = getSclOutDiffusionEigenfunctions( obj, varargin )
% GETSCLOUTDIFFUSIONEIGENFUNCTIONS Get scaled diffusion eigenfunctions 
% and Riemannian measure for the OS data of an nlsaModel_scl object
%
% Modified 2014/07/28

[ phi, mu, lambda ] = getEigenfunctions( getSclOutDiffusionOperator( obj ), ...
                                         varargin{ : } );
