function [ phi, mu, lambda ] = getOutDiffusionEigenfunctions( obj, varargin )
% GETDIFFUSIONEIGENFUNCTIONS Get diffusion eigenfunctions and Riemannian 
% measure for the OS data of an nlsaModel_err object
%
% Modified 2014/05/25

[ phi, mu, lambda ] = getEigenfunctions( getOutDiffusionOperator( obj ), ...
                                         varargin{ : } );
