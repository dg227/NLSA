function [ zeta, mu, gamma ] = getKoopmanEigenfunctions( obj, varargin )
% GETKOOPMANEIGENFUNCTIONS Get Koopman eigenfunctions, inner product weights, 
% and corresponding eigenvalues from an nlsaModel object.
%
% Modified 2020/04/15

[ zeta, mu, gamma ] = getEigenfunctions( getKoopmanOperator( obj ), ...
                                         varargin{ : } );
