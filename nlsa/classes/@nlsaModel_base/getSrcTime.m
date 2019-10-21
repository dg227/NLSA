function time = getSrcTime( obj )
% GETSRCTIME  Get source timestamps of nlsaModel_base object
%
% Modified 2013/11/08

time = obj.srcTime;
if numel( time ) == 1
    time = time{ 1 };
end
