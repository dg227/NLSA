function nD = getTrgDimension( obj, idxC )
% GETTRGDIMENSION  Get dimension of the target data of an nlsaModel_base object

% Modified 2014/01/09

if nargin == 1
    idxC = 1 : getNTrgComponent( obj ) ;
end

nD  = getDimension( obj.trgComponent( idxC, 1 ) ); 
