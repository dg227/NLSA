function comp = getDenEmbComponentQ( obj )
% GETDENEMBCOMPONENTQ Get embedded density data components (query partition) 
% of an nlsaModel_den object
%
% If denEmbComponentQ is empty, this function returns denEmbComponent
%
% Modified 2019/11/24

if isempty( obj.denEmbComponentQ )
    comp = obj.denEmbComponent;
else
    comp = obj.denEmbComponentQ; 
end
