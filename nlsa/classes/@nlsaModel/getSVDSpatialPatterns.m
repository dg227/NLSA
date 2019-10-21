function [ u, s ] = getSVDSpatialPatterns( obj, varargin )
% GETSVDSPATIALPATTERNS Get SVD spatial patterns and the corresponding 
% singular values for  an nlsaModel object
%
% Modified 2016/01/03

[ u, s ] = getSpatialPatterns( getLinearMap( obj ), ...
                               varargin{ : } );
