function obj = set( obj, varargin )
%
% SET Set properties of nlsaRoot objects. This function has the following two
%     types of calling syntax;
%
% obj = set( obj, Name, Value, ... ) assigns named properties to the specified
% values for all elements of the object array obj.
%
% obj = set( obj, propName, propVal ) assigns the properties specified in the
% cell array of strings propName to the corresponding values in the cell array
% propVal for all elements of the object array obj. The cell array propName 
% must be a vector of length n, but the cell array propVal can be m-by-n 
% where m is equal to length( obj ). set updates each object with the 
% associated set of values for the list of property names contained in 
% propName.
%
% Modified 2020/02/24
 
% Quick return if no properties to set
if isempty( varargin ) ...
    || ( nargin == 3 && isempty( varargin{ 1 } ) && isempty( varargin{ 2 } ) )
    return
end

% Check calling syntax
[ ifPropName, propName, propVal ] = isPropNameVal( varargin{ : } );

% Do recursive call if property name-value pairs are used
if ifPropName
    obj = set( obj, propName, propVal );
else
    propName = varargin{ 1 };
    propVal  = varargin{ 2 };
end

% Validate input arguments
if ~ifPropName && nargin > 3
    error( 'Invalid number of input arguments.' )
end

if ~iscellstr( propName ) || ~isvector( propName )
    propName
    error( 'Property names must be entered as a cell vector of strings.' )
end

if ~isrow( propName )
    propName = propName';
end

n = length( propName );
m = length( obj );

propObj = properties( obj );

if ~all( ismember( propName, propObj ) ) || numel( unique( propName ) ) < n
  error( 'Invalid or repeated property names.' )
end  

if ~iscell( propVal ) ||  size( propVal, 2 ) ~=  n ...
   || ( size( propVal, 1 ) ~= 1 && size( propVal, 1 ) ~= m )  
    error( 'Property values must be entered as a cell array with number of columns equal to the number of columns of the property name array and number of rows equal to 1 or the length of the input object array' )
end 

% Assign properties 
ifExpand = size( propVal, 2 ) < m;  
for iProp = 1 : n
    for iObj = 1 : m
        if ifExpand
            obj( iObj ) = setfield( obj( iObj ), ...
                propName{ iProp }, propVal{ 1, iProp } );
        else
            obj( iObj ) = setfield( obj( iObj ), ...
                propName{ iProp }, propVal{ iObj, iProp } );
        end
    end
end


