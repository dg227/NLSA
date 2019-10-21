function alpha = getAlpha( obj )
% GETALPHA  Get the normalization parameter alpha of nlsaDiffusionOperator 
% objects
%
% Modified 2014/01/29

alpha = zeros( size( obj ) );
for iObj = 1 : numel( obj )
    alpha( iObj ) = obj( iObj ).alpha;
end
