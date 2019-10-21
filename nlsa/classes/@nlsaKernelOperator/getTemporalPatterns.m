function v = getTemporalPatterns( obj, varargin )
% GETTEMPORALPATTERNS  Get temporal patterns of an nlsaKernelOperator object
%
% Modified 2015/09/22

v = getEigenfunctions( obj, varargin{ : } );
