function [ vT, mu, s ] = getSVDTemporalPatterns( obj, varargin )
% GETSVDTEMPORALPATTERNS Get SVD temporal patterns and Riemannian 
% measure of an nlsaModel object
%
% Modified 2015/10/21

[ vT, mu, s ] = getTemporalPatterns( getLinearMap( obj ), ...
                                     varargin{ : } );
