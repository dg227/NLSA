function time = getOutTime( obj )
% GETOUTTIME  Get out-of-sample timestamps of nlsaModel_ose object
%
% Modified 2019/07/07

time = obj.outTime;
if numel( time ) == 1
    time = time{ 1 };
end
