function S = filterField(R, names)
% Filter structure by fieldnames.
%
% Modified 2023/06/08

rNames = fieldnames(R);
keepNames = intersect(rNames, names);
rmNames = setdiff(rNames, keepNames);
S = rmfield(R, rmNames);
