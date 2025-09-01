function res = getNXB( obj )
% GETNXB  Returns the number of t<0 samples
%
% Modified 2013/12/03

res = zeros( size( obj ) );
for iObj = 1 : numel( obj )
    res( iObj ) = obj( iObj ).nXB;
end
