function file = getFile( obj, iF )
% GETDATAFILE Get files of an nlsaFilelist object 
%
% Modified 2014/03/31

if ~isscalar( obj )
    error( 'First argument must be a scalar.' )
end

if nargin == 1
    iF = 1 : getNFile( obj );
end

if isscalar( iF )
    file = obj.file{ iF };
else
    file = obj.file( iF );
end
