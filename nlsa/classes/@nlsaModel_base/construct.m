function obj2 = construct( obj1, varargin )
%% CONSTRUCT Generic constructor for nlsaComponent objects
%
% Modified 2019/11/13

n = numel( varargin );

switch numel( varargin )
case 0
    obj2 = obj1;
case 1
    if isa( varargin{ 1 }, 'nlsaComponent' )
        obj2 = duplicate( obj1, varargin{ 1 } );
    else
        error( 'Invalid input argument' )
    end
otherwise
    if iseven( n ) && ischar( varargin{ 1 } )
       obj2 = nlsaComponent( varargin{ : } ); 
   else
        error( 'Invalid input arguments' );
    end
end

