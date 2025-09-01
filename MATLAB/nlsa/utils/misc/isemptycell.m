function i = isemptycell( x )
% ISEMPTYCELL Returns true if the elements of the cell array x are empty
% 
% Modified 2014/12/15

i = cellfun( @isempty, x );
