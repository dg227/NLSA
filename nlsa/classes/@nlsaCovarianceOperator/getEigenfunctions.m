function [ v, mu, s ] = getEigenfunctions( obj, varargin )
% GETEIGENFUNCTIONS  Read right singular vectors of an nlsaCovarianceOperator 
% object
%
% Modified 2016/10/13

[ v, s ] = getRightSingularVectors( obj, varargin{ : } );

mu = ones( size( v, 1 ), 1 );
