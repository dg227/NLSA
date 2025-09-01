function den = getDensityKernelQ( obj, iCD )
% GETDENSITYKERNELQ Get kernel density of an  nlsaModel_den object, split into
% original partition from query partition.
%
% Modified 2020/01/28

if nargin == 1
    den = obj.densityQ;
else 
    den = obj.densityQ( iCD );
end
