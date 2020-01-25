function comp = getEmbComponentQ( obj )
% GETEMBCOMPONENTQ Get embedded source component (query partition) of an 
% nlsaModel_base object.
%
%  If embComponentQ is empty, getEmbComponentQ returns embComponent. 
%
% Modified 2020/01/25

if isempty( obj.embComponentQ )
    comp = obj.embComponent; 
else
    comp = obj.embComponentQ;
end
