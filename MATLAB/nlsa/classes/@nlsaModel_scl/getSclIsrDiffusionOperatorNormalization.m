function d = getSclIsrDiffusionOperatorNormalization( obj, varargin )
% GETSCLISRDIFFUSIONOPERATORNORMALIZATION Get normalization of the scaled
% ISR operator of an nlsaModel_scl object
%
% Modified 2014/07/28

d = getNormalization( getSclIsrDiffusionOperator( obj ), varargin{ : } );
