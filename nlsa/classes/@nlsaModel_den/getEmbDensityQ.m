function comp = getEmbDensityQ( obj )
% GETDENEMBDENSITYQ Get embedded density (query partition) of an nlsaModel_den 
% object
%
% If embDensityQ is empty, this function returns embDensity.
%
% Modified 2019/11/24

if isempty( obj.embDensityQ )
    comp = obj.embDensity;
else
    comp = obj.embDensityQ;
end
