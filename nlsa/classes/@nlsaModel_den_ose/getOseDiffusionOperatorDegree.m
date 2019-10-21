function d = getOseDiffusionOperatorDegree( obj, varargin )
% GETOSEDIFFUSIONOPERATORDEGREE Get degree of the OSE diffusion operator of 
% an nlsaModel_den_ose object
%
% Modified 2018/07/01

d = getDegree( getOseDiffusionOperator( obj ), varargin{ : } );
