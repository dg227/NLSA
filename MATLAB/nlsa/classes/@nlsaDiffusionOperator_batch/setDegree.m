function setDegree( obj, d, iB, varargin )
% SETDEGREE  Set degree batch data of an 
% nlsaDiffusionOperator_batch object 
%
% Modified 2016/01/26

partition = getPartition( obj );

if ~ischar( varargin{ 1 } )
    iR = varargin{ 1 };  % varargin{ 1 } stores realization
    varargin = varargin( 2 : end );
else
    [ iB, iR ] = gl2loc( partition, iB );
end

if ~isvector( d ) || numel( d ) ~= getBatchSize( partition( iR ), iB )
    error( 'Incompatible number of elements in degree vector' )
end
if isrow( d )
    d = d';
end


file = fullfile( getDegreePath( obj ), ... 
                 getDegreeFile( obj, iB, iR ) );
save( file, 'd', varargin{ : } )

