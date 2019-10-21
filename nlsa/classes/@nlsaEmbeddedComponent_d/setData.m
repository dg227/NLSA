function setData( obj, xENorm2, xE, iB, varargin )
% SETDATA  Write differenced data for an nlsaEmbeddedComponent_xi_d object
%
% Modified 2014/07/10

[ nDE, nSBE ] = getBatchArraySize( obj, iB );

ifSave = false( 1, 3 );

if ~isempty( xENorm2 )
    if ~isvector( xENorm2 ) || numel( xENorm2 ) ~= nSBE
        error( 'Incompatible size of state eror norm array' )
    end
    if iscolumn( xENorm2 )
        xENorm2 = xENorm2';
    end
    ifSave( 1 ) = true;
end

if ~isempty( xE )
    if any( size( xE ) ~= [ nDE nSBE ] )
        error( 'Incompatible size of state error array' )
    end
    ifSave( 2 ) = true;
end
    
    
file    = fullfile( getDataPath( obj ), ... 
                    getDataFile( obj, iB ) );
varNames = { 'xENorm2' 'xE' };
save( file, varNames{ ifSave }, varargin{ : } )

