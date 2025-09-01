function obj = setFile( obj, file )
% SETFILE Set filenames of an nlsaFilelist object
%
% Modified 2014/04/07

if isrowstr( file )
    file = { file };
end

if isscalar( obj )

    nF = getNFile( obj );

    if nF == 1 && isrowstr( file )
        obj.file = { file };
    end

    if ~iscellstr( file ) ...
      || ~isrow( file ) ...
      || numel( file ) ~= nF
        error( 'Invalid filename specification' )
    end

    obj.file = file;

else

    for iObj = 1 : numel( obj )
        obj( iObj ) = setFile( obj( iObj ), file{ iObj } );
    end

end
