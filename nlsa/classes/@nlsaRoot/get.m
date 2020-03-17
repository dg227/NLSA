function propVal = get( obj, propName )
%
% GET Get properties of nlsaRoot objects. 
%
% obj = get( obj, propName ) returns the values of the properties specified in
% the cell array of strings propName for the object array obj. propName must
% be a length-n vector, where n is the number of property names. propVal is a 
% cell array of size m-by-n where m is equal to length( obj ). If n = 1, 
% propName can be set to a string. If m = n = 1, propVal returns the actual 
% value of propName, as opposed to a cell array containing that value. 
% 
% Modified 2020/02/24

% Check calling syntax
if ischar( propName )
    propName = { propName };
end

if ~iscellstr( propName ) || ~isvector( propName )
    error( 'Property names must be specified as a character string or as a cell vector of strings' ); 
end

if ~isrow( propName )
    propName = propName';
end

if ~all( ismember( propName, properties( obj ) ) )
    error( 'Invalid property names' )
end

n = length( propName );
m = length( obj );

propVal = cell( m, n );

for iProp = 1 : n
    for iObj = 1 : m
        propVal{ iObj, iProp } = getfield( obj( iObj ), propName{ iProp } );
    end
end

if isscalar( propVal )
    propVal = propVal{ 1 };
end
