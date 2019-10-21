function nSB = getBatchSize( obj, iB )
% GETBATCHSIZE  Get batch sizes of an nlsaEmbeddedComponent object 
%
% Modified 2014/04/14

if ~isscalar( obj )
    error( 'First argument must be a scalar nlsaComponent object.' )
end

nB = getNBatch( obj );
if nargin == 1
    iB = 1 : nB;
end

nSB = zeros( size( iB ) );

ifMain   = iB >= 1 & iB <= nB;
ifBefore = iB == 0;
ifAfter  = iB == nB + 1;
if nnz( ifMain | ifBefore | ifAfter ) ~= numel( iB )
    error( 'Out of range batch specification' )
end


nSB( ifMain )   = getBatchSize@nlsaComponent( obj, iB( ifMain ) );
nSB( ifBefore ) = getNXB( obj );
nSB( ifAfter )  = getNXA( obj );
