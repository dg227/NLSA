function q = getData( obj, varargin )
% GETDATA Read density data of an nlsaKernelDensity_ose_fb object
%
% Modified 2018/07/08

q = getDensity( obj, varargin{ : } );
q = q';
