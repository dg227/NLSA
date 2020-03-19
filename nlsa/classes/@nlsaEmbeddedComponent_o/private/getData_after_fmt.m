function x = getData_after_fmt( obj, outFormat )
% GETDATA_AFTER_FMT Read data after main interval from an 
% nlsaEmbeddedComponent_o object in overlap format
%
% Modified 2020/03/18

if ~isscalar( obj )
    error( 'First input argument must be a scalar nlsaEmbeddedComponent_o object' )
end

nB = getNBatch( obj );

x = getData_fmt( obj, nB + 1, outFormat );
