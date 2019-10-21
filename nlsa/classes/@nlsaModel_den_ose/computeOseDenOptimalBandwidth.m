function [ epsilonOpt, Info ] = computeOseDenOptimalBandwidth( obj )
% COMPUTEOSEDENOPTIMALBANDWIDTH Compute optimal bandwidth for kernel density
% estimation forf an  nlsaModel_den_ose object
%
% Modified 2018/07/04

[ epsilonOpt, Info ] = computeOptimalBandwidth( getOseDensityKernel( obj ) );
