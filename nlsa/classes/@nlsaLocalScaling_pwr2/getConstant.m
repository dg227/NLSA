function [ bX, bXi, c ] = getConstant( obj )
% GETCONSTANT  Get proportionality constants of an nlsaLocalScaling_pwr object
%
% Modified 2014/06/25

[ bX, bXi ] = getConstant@nlsaLocalScaling_exp( obj );
c   = obj.c;
