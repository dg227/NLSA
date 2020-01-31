function [ tst, props ] = isPropValName( varargin )
%% ISPROPVALNAME Check that input arguments form a list of property name-value
% pairs.
%
% Modified 2019/11/16

if ~iseven( nargin )
    tst = false;
    props = {};
    return
end

tmp = varargin( 1 : 2 : end );
if ~iscellstr( tmp ) || numel( unique( tmp ) ) < numel( tmp )
    tst = false;
    props = {};
    return
end

tst = true;
props = tmp;

