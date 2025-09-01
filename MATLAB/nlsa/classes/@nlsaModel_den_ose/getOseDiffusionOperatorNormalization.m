function d = getOseDiffusionOperatorNormalization( obj, varargin )
% GETOSEDIFFUSIONOPERATORNORMALIZATION Get normalization of the OSE diffusion 
% operator of an nlsaModel_den_ose object
%
% Modified 2018/07/01

d = getNormalization( getOseDiffusionOperator( obj ), varargin{ : } );
