function obj2 = construct( obj1, src, varargin )
%% CONSTRUCT Generic constructor for nlsaComponent objects
%
% Modified 2019/11/13


if nargin >= 2 && isa( src, 'nlsaComponent' ) && isvector( obj1 ) ...
    && ismatrix( src )
    obj2 = repvec( obj1, src, varargin{ : } );
    return
end

obj2 = construct@nlsaRoot( obj1, src, varargin );

