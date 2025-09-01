function setVelocity( obj, xiENorm2, xiE, xiNorm2, iB, varargin )
% SETVELOCITY  Write phase space velocity difference data for an 
% nlsaEmbeddedComponent_xi_d object
%
% Modified 2014/07/10

[ nDE, nSBE ] = getBatchArraySize( obj, iB );

ifSave = false( 1, 3 );

if ~isempty( xiENorm2 )
    if ~isvector( xiENorm2 ) || numel( xiENorm2 ) ~= nSBE
        error( 'Incompatible size of phase space velocity eror norm array' )
    end
    if iscolumn( xiENorm2 )
        xiENorm2 = xiENorm2';
    end
    ifSave( 1 ) = true;
end

if ~isempty( xiE )
    if any( size( xiE ) ~= [ nDE nSBE ] )
        error( 'Incompatible size of phase space velocity array' )
    end
    ifSave( 2 ) = true;
end
    
if ~isempty( xiNorm2 )
    if ~isvector( xiNorm2 ) || numel( xiNorm2 ) ~= nSBE 
        error( 'Incompatible size of reference phase space velocity norm array' )
    end
    if iscolumn( xiNorm2 )
        xiNorm2 = xiNorm2';
    end
    ifSave( 3 ) = true;
end
    
file    = fullfile( getVelocityPath( obj ), ... 
                    getVelocityFile( obj, iB ) );
varNames = { 'xiENorm2' 'xiE' 'xiNorm2' };
save( file, varNames{ ifSave }, varargin{ : } )
