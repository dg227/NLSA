function path = getOutDiffusionOperatorPath( obj )
% GETOUTDIFFUSIONOPERATORPATH  Get diffusion operator path for the OS data 
% of an nlsaModel_err object
%
% Modified 2014/05/25

path = getPath( getOutDiffusionOperator( obj ) );
