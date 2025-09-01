function d = getSclIsrDiffusionOperatorDegree( obj, varargin )
% GETPRJDIFFUSIONOPERATORDEGREE Get degree of the scaled ISR operator
% of an nlsaModel_scl object
%
% Modified 2014/07/28

d = getDegree( getSclIsrDiffusionOperator( obj ), varargin{ : } );
