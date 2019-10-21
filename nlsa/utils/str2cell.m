function cl = str2cell( str, n )

if ~ischar( str )
    error( 'First input argument must be a character array' )
end

cl = cell( 1, n );
for i = 1 : n
    cl{ i } = str;
end

