function d = getOseDiffusionOperatorNormalization( obj, varargin )
% GETOSEDIFFUSIONOPERATORNORMALIZATION Get normalization of the OSE diffusion 
% operator of an nlsaModel_ose object
%
% Modified 2014/04/22

d = getNormalization( getOseDiffusionOperator( obj ), varargin{ : } );
