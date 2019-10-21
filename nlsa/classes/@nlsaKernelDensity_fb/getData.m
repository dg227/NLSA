function q = getData( obj, varargin )
% GETDATA Read density data of an nlsaKernelDensity_fb object
%
% Modified 2015/10/28

q = getDensity( obj, varargin{ : } );
q = q';
