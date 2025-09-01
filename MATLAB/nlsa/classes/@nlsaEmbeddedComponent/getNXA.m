function res = getNXA( obj )
% GETNXA  Returns the number of t>T samples
%
% Modified 2013/12/03

res = zeros( size( obj ) );
for iObj = 1 : numel( obj )
    res( iObj ) = obj( iObj ).nXA;
end
