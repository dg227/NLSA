function strJ = strjoin_e( str, varargin )
%STRJOIN  Join cell array of strings into single string ignoring emtpy strings
%  
% Modified 2014/12/15

ifEmpty = isemptycell( str );
strJ = strjoin( str( ~ifEmpty ), varargin{ : } );
