function nN = getNNeighborsTest( obj )
% GETNNEIGHBORSTEST  Get number of nearest neighrors for the test data of 
% an nlsaDiffusionOperator_batch object
%
% Modified 2016/01/25

if ~isempty( obj.nNT )
    nN = obj.nNT;
else
    nN = getNNeighbors( obj );
end
