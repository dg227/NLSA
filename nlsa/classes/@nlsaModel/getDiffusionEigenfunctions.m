function [ phi, mu, lambda ] = getDiffusionEigenfunctions( obj, varargin )
% GETDIFFUSIONEIGENFUNCTIONS Get diffusion eigenfunctions and Riemannian 
% measure of an nlsaModel object
%
% Modified 2014/04/16

[ phi, mu, lambda ] = getEigenfunctions( getDiffusionOperator( obj ), ...
                                         varargin{ : } );
