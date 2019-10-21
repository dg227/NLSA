function [ l, d, q ] = getOperator( obj )
% GETOPERATOR  Read operator data of an nlsaDiffusionOperator_gl object
%
% Modified 2014/07/16


varNames = { 'l' 'd' 'q' };
file = fullfile( getOperatorPath( obj ), getOperatorFile( obj ) );
load( file, varNames{ 1 : nargout } )
