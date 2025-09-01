function obj2 = duplicate( obj1, src, varargin )
%% DUPLICATE Duplicate nlsaRoot object
%
% varargin contains optional input arguments specifying property names/values 
% that are not taken from src. These arguments must have the appropriate 
% syntax for the set function of nlsaRoot objects.
%
% Modified 2020/02/25

% Check input properties

props = properties( obj1 );
if ~all( ismember( properties( src ), props ) )
    error( 'Incompatible properties of input arguments' )
end

% Check input array sizes for compatibility, and expand singleton dimensions

sizObj1  = size( obj1 );
sizSrc   = size( src );
nSizObj1 = numel( sizObj1 );
nSizSrc  = numel( sizSrc );

if nSizObj1 <= nSizSrc
    sizObj2 = [ sizObj1 sizSrc( nSizObj1 + 1 : end ) ];
    sizSrc2 = sizSrc;
end

if nSizObj1 > nSizSrc
    sizSrc2 = [ sizSrc1 sizObj1( nSizSrc1 : end ) ];
    sizObj2 = sizObj1;
end

ifCheck = sizObj2 ~= sizSrc2;

if ~all( sizObj2( ifCheck ) == 1 | sizSrc2( ifCheck ) == 1 );
    error( 'Incompatible input array dimensions' )
end

ifRepObj = ifCheck & ( sizObj2 == 1 );
nRepObj = ones( 1, numel( sizObj2 ) );
nRepObj( ifRepObj ) = sizSrc2( ifRepObj ); 
obj2 = repmat( obj1, nRepObj );

ifRepSrc = ifCheck & ( sizSrc2 == 1 ); 
nRepSrc = ones( 1, numel( sizSrc2 ) );
nRepSrc( ifRepSrc ) = sizObj2( ifRepSrc );
src2 = repmat( src, nRepSrc );

% If no optional inputs are passed, assign the properties of obj2 by 
% dublication of the properties of src2
if isempty( varargin )
    obj2 = set( obj2, props, get( src2, props ) )
    return
end

% Validate optional input arguments
[ tst, props2 ] = isPropNameVal( varargin{ : } );
if ~tst 
    error( 'Optional input arguments must be passed as property name-value pairs.' )
end

% Duplicate only the properties that are not in props2
props = setdiff( props, props2 );
obj2 = set( obj2, props, get( src2, props ) );

% Set the remaining properties
for iProp = 1 : numel( props2 )
    obj2 = set( obj2, varargin{ 2 * iProp - 1 }, varargin{ 2 * iProp } );
end


