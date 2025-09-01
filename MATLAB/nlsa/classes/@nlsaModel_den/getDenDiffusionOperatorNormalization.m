function d = getDenDiffusionOperatorNormalization( obj, varargin )
% GETDENDIFFUSIONOPERATORNORMALIZATION Get normalization of the diffusion 
% operator for the diffusion operator of an nlsaModel_den object
%
% Modified 2015/01/03

d = getNormalization( getDenDiffusionOperator( obj ), varargin{ : } );
