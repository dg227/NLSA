function nD = getDimension( obj, idxC )
% GETDIMENSION  Get dimension of the source data of an nlsaModel_base object

% Modified 2014/01/09

if nargin == 1
    idxC = 1 : getNSrcComponent( obj ) ;
end

nD = getDimension( obj.srcComponent( idxC, 1 ) ); 
