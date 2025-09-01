function [ tst, props, vals ] = isPropNameVal( varargin )
%% ISPROPNAMEVAL Check that input arguments form a list of property name-value
% pairs.
%
% Modified 2020/02/23

if ~iseven( nargin )
    tst = false;
    props = {};
    vals  = {};
    return
end

tmp = varargin( 1 : 2 : end );
if ~iscellstr( tmp ) || numel( unique( tmp ) ) < numel( tmp )
    tst = false;
    props = {};
    vals = {};
    return
end

tst = true;
props = tmp;
vals = varargin( 2 : 2 : end ); 


