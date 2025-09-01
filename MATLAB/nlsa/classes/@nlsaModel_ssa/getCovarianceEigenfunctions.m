function [ phi, mu, sigma ] = getCovarianceEigenfunctions( obj, varargin )
% GETCOVARIANCEEIGENFUNCTIONS Get covariance eigenfunctions and singular values 
% measure of an nlsaModel_ssa object
%
% Modified 2016/06/13

[ phi, mu, sigma ] = getEigenfunctions( getCovarianceOperator( obj ), ...
                                       varargin{ : } );
