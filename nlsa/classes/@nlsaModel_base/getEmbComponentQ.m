function comp = getEmbComponentQ( obj )
% GETEMBCOMPONENTQ Get embedded source component (query partition) of an 
% nlsaModel_base object.
%
%  If embComponentQ is empty, getEmbComponentQ returns embComponent. 
%
% Modified 2019/11/24

if isempty( obj.embComponentQ )
    comp = obj.embComponentQ; 
else
    comp = obj.embComponent;
end
