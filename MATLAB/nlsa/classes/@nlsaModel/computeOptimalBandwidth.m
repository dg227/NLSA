function [ epsilonOpt, Info ] = computeOptimalBandwidth( obj )
% COMPUTEOPTIMALBANDWIDTH Compute optimal bandwidth for the diffusion operator
% in an nlsaModel_den object
%
% Modified 2015/12/17

[ epsilonOpt, Info ] = computeOptimalBandwidth( getDiffusionOperator( obj ) );
