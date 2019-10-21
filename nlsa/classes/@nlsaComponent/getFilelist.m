function file = getFilelist( obj )
% GETFILELIST  Get filelists of nlsaComponent objects
%
% Modified 2014/04/04

for iObj = numel( obj )  : -1 : 1 
    file( iObj ) = obj( iObj ).file;
end
file = reshape( file, size( obj ) );
