function file = getDataFile( obj, iB )
% GETDATAFILE Get data files of an nlsaEmbeddedComponent object
%
% Modified 2014/04/06

if ~isscalar( obj )
    error( 'First argument must be a scalar nlsaEmbeddedComponent object.' )
end

nB = getNBatch( obj );
if nargin == 1
    iB = 1 : nB;
end

file = cell( size( iB ) );

ifMain   = iB >= 1 & iB <= nB;
ifBefore = iB == 0;
ifAfter  = iB == nB + 1;
if nnz( ifMain | ifBefore | ifAfter ) ~= numel( iB )
    error( 'Out of range batch specification' )
end

if isscalar( iB( ifMain ) )
    file( ifMain )   = { getDataFile@nlsaComponent( obj, iB( ifMain ) ) };
else
    file( ifMain )   = getDataFile@nlsaComponent( obj, iB( ifMain ) );
end
file( ifBefore ) = { getDataFile_before( obj ) };
file( ifAfter )  = { getDataFile_after( obj ) };

if isscalar( file )
    file = file{ 1 };
end
