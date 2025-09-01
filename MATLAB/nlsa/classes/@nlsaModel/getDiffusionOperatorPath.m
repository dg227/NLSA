function path = getDiffusionOperatorPath( obj )
% GETDIFFUSIONOPERATORPATH  Get diffusion operator path of an nlsaModel object
%
% Modified 2014/04/10

path = getPath( getDiffusionOperator( obj ) );
