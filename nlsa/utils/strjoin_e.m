function strJ = strjoin_e( str, varargin )
%STRJOIN  Join cell array of strings into single string ignoring emtpy strings
%  
% Modified 2021/01/25

if ischar( str )
    str = { str };
end
ifEmpty = isemptycell( str );
strJ = strjoin( str( ~ifEmpty ), varargin{ : } );
