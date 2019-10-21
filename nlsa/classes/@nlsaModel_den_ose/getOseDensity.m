function den = getOseDensity( obj )
% GETOSEDENSITY Get OSE density of an  nlsaModel_den_ose object
%
% Modified 2015/12/17

den = getDensity( getOseDensityKernel( obj ) );
