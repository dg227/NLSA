function obj2 = duplicate( obj1, src, varargin )
%% DUPLICATE Duplicate nlsaRoot object
%
% Modified 2019/11/13

% Check input sizes
if ~isequal( size( obj1 ), size( src ) )
    error( 'Incompatible sizes of input arguments.' )
end

% Check input properties
props = properties( obj1 );
if ~isempty( setdiff( props, properties( src ) ) )
    error( 'Incompatible properties of input arguments' )
end

if isempty( varargin )
    obj2 = obj1;
    set( obj2, props, get( src, props ) )
    return;
end

[ tst, props2 ] = isPropNameVal( varargin );
if ~tst
    error( 'Optional input arguments must be entered as property name-value pairs' )
end

props = setdiff( props, props2 );
set( obj2, props, get( src, props ) )
set( obj2, props2, varargin( 2 : 2 : end ) );


