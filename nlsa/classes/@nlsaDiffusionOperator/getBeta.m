function beta = getBeta( obj )
% GETBETA  Get the normalization parameter beta of nlsaDiffusionOperator 
% objects
%
% Modified 2021/02/27

beta = zeros( size( obj ) );
for iObj = 1 : numel( obj )
    beta( iObj ) = obj( iObj ).beta;
end
