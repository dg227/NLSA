function d = getOseDenDiffusionOperatorNormalization( obj, varargin )
% GETOSEDENDIFFUSIONOPERATORNORMALIZATION Get normalization of the diffusion 
% operator for the OSE diffusion operator of an nlsaModel_den_ose object
%
% Modified 2015/01/03

d = getNormalization( getOseDenDiffusionOperator( obj ), varargin{ : } );
