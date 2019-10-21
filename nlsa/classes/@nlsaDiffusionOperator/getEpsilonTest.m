function epsilon = getEpsilonTest( obj )
% GETEPSILONTEST  Get the kernel bandwidth parameter for the test data of 
% nlsaDiffusionOperator objects
%
% Modified 2016/01/25

epsilon = zeros( size( obj ) );
for iObj = 1 : numel( obj )
    if ~isempty( obj( iObj ).epsilonT )
        epsilon( iObj ) = obj( iObj ).epsilonT;
    else
        epsilon( iObj ) = getEpsilon( obj( iObj ) );
    end
end
