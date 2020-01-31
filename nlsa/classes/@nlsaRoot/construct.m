function obj2 = construct( obj1, src, varargin )
%% CONSTRUCT Generic constructor for NLSA objects
%
% Modified 2019/11/13

switch nargin
case 0
    obj2 = nlsaRoot();
case 1
    obj2 = obj1;
otherwise
    obj2 = duplicate( obj1, src, varargin{ : } );
end

