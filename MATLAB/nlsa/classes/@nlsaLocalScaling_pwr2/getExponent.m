function [ pX, pXi, p ] = getExponent( obj )
% GETEXPONENT  Get exponents of nlsaLocalScaling_pwr object
%
% Modified 2014/06/18

[ pX, pXi ] = getExponent@nlsaLocalScaling_exp( obj );
p   = obj.p;
