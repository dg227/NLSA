function den = getDensityKernel( obj, iCD )
% GETDENSITYKERNEL Get kernel density of an  nlsaModel_den object
%
% Modified 2019/11/05

if nargin == 1
    den = obj.density;
else 
    den = obj.density( iCD );
end
