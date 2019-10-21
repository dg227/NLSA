function setOperator( obj, pVal, pInd, iB, varargin )
% SETOPERATOR  Set operator batch data of an nlsaDiffusionOperator_batch object 
%
% Modified 2014/05/01

sizPVal = size( pVal );
sizPInd = size( pInd );


if   numel( sizPVal ) ~= numel( sizPInd ) ...
  || any( sizPVal ~= sizPInd )
    error( 'Incompatible operator and nearest neighbor index data' )
end

partion = getPartition( obj );

if ~ischar( varargin{ 1 } )
    iR = varargin{ 1 };  % varargin{ 1 } stores realization
    varargin = varargin( 2 : end );
else
    [ iB, iR ] = gl2loc( partition, iB );
end

[ sizB( 1 ), sizB( 2 ) ] = getBatchArraySize( obj, iB, iR );
if any( sizPVal ~= sizB )
    error( 'Incompatible size of operator data array' )
end

file = fullfile( getOperatorPath( obj ), ... 
                 getOperatorFile( obj,  iB, iR ) );
save( file, 'pVal', 'pInd', varargin{ : } )

