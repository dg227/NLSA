function time = getTrgTime( obj )
% GETTRGTIME  Get target timestamps of nlsaModel_base object
%
% Modified 2014/07/22

time = obj.trgTime;
if isscalar( time )
    time = time{ 1 };
end
