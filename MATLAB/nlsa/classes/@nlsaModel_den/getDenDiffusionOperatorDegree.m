function d = getDenDiffusionOperatorDegree( obj, varargin )
% GETDENDIFFUSIONOPERATORDEGREE Get degree of diffusion operator for the 
% diffusion operator of an nlsaModel_den object
%
% Modified 2015/01/03

d = getDegree( getDenDiffusionOperator( obj ), varargin{ : } );
