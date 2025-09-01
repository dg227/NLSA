function d = getIsrDiffusionOperatorNormalization( obj, varargin )
% GETISRDIFFUSIONOPERATORNORMALIZATION Get normalization of the 
% model -> nature OSE diffusion operator of an nlsaModel_err object
%
% Modified 2014/07/28

d = getNormalization( getIsrDiffusionOperator( obj ), varargin{ : } );
