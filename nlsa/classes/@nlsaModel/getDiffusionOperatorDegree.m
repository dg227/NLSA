function d = getDiffusionOperatorDegree( obj, varargin )
% GETDIFFUSIONOPERATORDEGREE Get degree of diffusion operator of an 
% nlsaModel object
%
% Modified 2014/02/01

d = getDegree( getDiffusionOperator( obj ), varargin{ : } );
