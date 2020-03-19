function x = getData_before_fmt( obj, outFormat )
% GETDATA_BEFORE_FMT Read data before main interval from an 
% nlsaEmbeddedComponent_e object in overlap format
%
% Modified 2020/03/19

if ~isscalar( obj )
    error( 'First input argument must be a scalar nlsaEmbeddedComponent_o object' )
end

x = getData_fmt( obj, 0, outFormat );
