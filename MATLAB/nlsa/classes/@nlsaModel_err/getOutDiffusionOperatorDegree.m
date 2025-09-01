function d = getOutDiffusionOperatorDegree( obj, varargin )
% GETOUTDIFFUSIONOPERATORDEGREE Get degree of diffusion operator for the OS 
% data of an nlsaModel_err object
%
% Modified 2015/05/25

d = getDegree( getOutDiffusionOperator( obj ), varargin{ : } );
