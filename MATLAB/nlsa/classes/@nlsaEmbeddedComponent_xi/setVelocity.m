function setVelocity( obj, xiNorm2, xi, iB, varargin )
% SETVELOCITY  Write phase space velocity data of an nlsaEmbeddedComponent_xi 
% object
%
% varargin is used to pass flags to Matlab's save routine.
%
% Modified 2014/04/05

[ nDE, nSBE ] = getBatchArraySize( obj, iB );

ifSave = false( 1, 2 );
if ~isempty( xiNorm2 )
    if ~isvector( xiNorm2 ) || numel( xiNorm2 ) ~= nSBE
        error( 'Incompatible size of phase space velocity norm array' )
    end
    if iscolumn( xiNorm2 )
        xiNorm2 = xiNorm2';
    end
    ifSave( 1 ) = true;
end

if ~isempty( xi )
    if any( size( xi ) ~= [ nDE nSBE ] )
        error( 'Incompatible size of phase space velocity array' )
    end
    ifSave( 2 ) = true;
end

file     = fullfile( getVelocityPath( obj ), ... 
                     getVelocityFile( obj, iB ) );
varNames = { 'xiNorm2' 'xi' };
save( file, varNames{ ifSave }, varargin{ : } )

