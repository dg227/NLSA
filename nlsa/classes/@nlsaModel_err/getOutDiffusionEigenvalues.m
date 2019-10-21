function lambda = getOutDiffusionEigenvalues( obj )
% GETMODDIFFUSIONEIGENVALUES Get diffusion eigenvalues for the OS data of an 
% nlsaModel_out object
%
% Modified 2014/05/24

lambda = getEigenvalues( getOutDiffusionOperator( obj ) );

