function epsilon = getEpsilon( obj )
% GETEPSILON  Get the kernel bandwidth of nlsaKernelDensity objects
%
% Modified 2018/07/05

epsilon = zeros( size( obj ) );
for iObj = 1 : numel( obj )
    epsilon( iObj ) = obj( iObj ).epsilon;
end
