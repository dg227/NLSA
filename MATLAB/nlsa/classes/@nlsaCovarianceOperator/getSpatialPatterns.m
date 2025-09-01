function [ u, s ] = getSpatialPatterns( obj, iC, varargin );
% GETSPATIALPATTERNS  Compute spatial patterns of an nlsaCovarianceOperator\
% object
%
% Modified 2016/01/03

Opt.ifS   = true; % scale spatial pattens by the singular values
Opt = parseargs( Opt, varargin{ : } );

[ u, s ] = getLeftSingularVectors( obj, iC );
if Opt.ifS
    u = bsxfun( @times, u, s' );
end
