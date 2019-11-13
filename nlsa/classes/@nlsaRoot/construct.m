function obj2 = construct( obj1 )
%% CONSTRUCT Generic constructor for NLSA objects
%
% Modified 2019/11/13

switch nargin
case 0
    obj2 = nlsaRoot();
case 1
    if isa( obj1, 'nlsaRoot' )
        obj2 = obj1;
    else
        error( 'Invalid input argument' )
    end
otherwise
    error( 'Invalid number of input arguments' );
end

