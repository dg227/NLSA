function [ vT, mu ] = computeEigenfunctions( obj, kOp, varargin )
% COMPUTEEIGENFUNCTIONS Compute right (temporal) patterns associated with 
% nlsaLinearMap objects
% 
% Modified 2014/08/05

[ vT, mu ] = computeRightPatterns( obj, kOp, varargin{ : } );

