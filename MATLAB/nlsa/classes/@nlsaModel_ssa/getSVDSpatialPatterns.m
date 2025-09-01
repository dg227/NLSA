function [ u, s ] = getSVDSpatialPatterns( obj, varargin )
% GETSVDSPATIALPATTERNS Get SVD spatial patterns and the corresponding 
% singular values for  an nlsaModel_ssa object
%
% Modified 2016/05/31

[ u, s ] = getSpatialPatterns( getLinearMap( obj ), ...
                               varargin{ : } );
