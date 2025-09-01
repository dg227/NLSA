function epsilon = getEpsilon( obj )
% GETEPSILON  Get the kernel bandwidth of nlsaDiffusionOperator objects
%
% Modified 2014/01/29

epsilon = zeros( size( obj ) );
for iObj = 1 : numel( obj )
    epsilon( iObj ) = obj( iObj ).epsilon;
end
