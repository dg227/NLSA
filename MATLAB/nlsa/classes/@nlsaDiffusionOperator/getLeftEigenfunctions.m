function [ v, mu, lambda ] = getLeftEigenfunctions( obj, varargin )
% GETEIGENFUNCTIONS  Read left eigenfunction data of an 
% nlsaDiffusionOperator object.
%
% This function assumes that obj is a self-adjoing operator, so left and right
% eigenfunctions coincide.
%
% Modified 2020/08/07

[ v, mu, lambda ] = getEigenfunctions( obj, varargin{ : } ); 

