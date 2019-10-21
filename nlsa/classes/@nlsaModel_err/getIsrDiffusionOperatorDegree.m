function d = getIsrDiffusionOperatorDegree( obj, varargin )
% GETISRDIFFUSIONOPERATORDEGREE Get degree of the ISR 
% diffusion operator of an nlsaModel_err object
%
% Modified 2014/07/28

d = getDegree( getIsrDiffusionOperator( obj ), varargin{ : } );
