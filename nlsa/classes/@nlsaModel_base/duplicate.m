function obj2 = duplicate( obj1, src )
%% DUPLICATE Duplicate nlsaComponent object
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

obj2 = obj1;
set( obj2, props, get( src, props ) )

