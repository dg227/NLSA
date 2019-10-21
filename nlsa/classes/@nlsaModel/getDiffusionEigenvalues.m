function lambda = getDiffusionEigenvalues( obj )
% GETDIFFUSIONEIGENVALUES Get diffusion eigenvalues of an nlsaModel object
%
% Modified 2014/02/12

lambda = getEigenvalues( getDiffusionOperator( obj ) );

