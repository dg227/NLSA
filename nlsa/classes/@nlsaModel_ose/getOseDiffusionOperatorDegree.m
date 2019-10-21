function d = getOseDiffusionOperatorDegree( obj, varargin )
% GETOSEDIFFUSIONOPERATORDEGREE Get degree of the OSE diffusion operator of 
% an nlsaModel_ose object
%
% Modified 2014/04/22

d = getDegree( getOseDiffusionOperator( obj ), varargin{ : } );
