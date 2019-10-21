function den = getDensity( obj )
% GETDENSITY Get density of an  nlsaModel_den object
%
% Modified 2015/12/17

den = getDensity( getDensityKernel( obj ) );
