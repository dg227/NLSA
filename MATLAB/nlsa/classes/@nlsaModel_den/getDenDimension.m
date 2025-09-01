function nD = getDenDimension( obj, idxC )
% GETiDENDIMENSION  Get dimension of the density data of an nlsaModel_den object

% Modified 2014/12/30

if nargin == 1
    idxC = 1 : getNDenComponent( obj ) ;
end

nD = getDimension( obj.denComponent( idxC, 1 ) ); 
