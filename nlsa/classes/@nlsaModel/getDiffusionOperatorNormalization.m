function d = getDiffusionOperatorNormalization( obj, varargin )
% GETDIFFUSIONOPERATORNORMALIZATION Get normalization of the diffusion 
% operator of an nlsaModel object
%
% Modified 2015/01/03

d = getNormalization( getDiffusionOperator( obj ), varargin{ : } );
