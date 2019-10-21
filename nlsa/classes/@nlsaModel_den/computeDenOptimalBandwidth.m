function [ epsilonOpt, Info ] = computeDenOptimalBandwidth( obj )
% COMPUTEDENOPTIMALBANDWIDTH Compute optimal bandwidth for kernel density
% estimation forf an  nlsaModel_den object
%
% Modified 2016/02/01

[ epsilonOpt, Info ] = computeOptimalBandwidth( getDensityKernel( obj ) );
