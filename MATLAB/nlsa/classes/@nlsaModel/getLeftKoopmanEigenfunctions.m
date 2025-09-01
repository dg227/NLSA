function [ zeta, mu, gamma ] = getLeftKoopmanEigenfunctions( obj, varargin )
% GETKOOPMANLEFTEIGENFUNCTIONS Get left Koopman eigenfunctions, inner product 
% weights, and corresponding eigenvalues from an nlsaModel object.
%
% Modified 2020/08/28

[ zeta, mu, gamma ] = getLeftEigenfunctions( getKoopmanOperator( obj ), ...
                                             varargin{ : } );
