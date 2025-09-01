function den = getDensity( obj, iCD )
% GETDENSITY Get density of an  nlsaModel_den object
%
% Modified 2019/11/06

if nargin == 1
    iCD = 1;
end

den = getDensity( getDensityKernel( obj, 1 ) );
