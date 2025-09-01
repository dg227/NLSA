function [ phi, mu, lambda ] = getDiffusionEigenfunctions( obj, varargin )
% GETDIFFUSIONEIGENFUNCTIONS Get diffusion eigenfunctions, inner product
% weights, and corresponding eigenvalues from an nlsaModel object.
%
% Modified 2014/04/16

[ phi, mu, lambda ] = getEigenfunctions( getDiffusionOperator( obj ), ...
                                         varargin{ : } );
