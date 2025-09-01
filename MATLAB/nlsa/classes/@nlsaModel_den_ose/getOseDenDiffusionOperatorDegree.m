function d = getOseDenDiffusionOperatorDegree( obj, varargin )
% GETDENDIFFUSIONOPERATORDEGREE Get degree of diffusion operator for the OSE 
% diffusion operator of an nlsaModel_den object
%
% Modified 2018/07/04

d = getDegree( getOseDenDiffusionOperator( obj ), varargin{ : } );
