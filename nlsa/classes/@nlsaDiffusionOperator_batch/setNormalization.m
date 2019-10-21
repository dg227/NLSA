function setNormalization( obj, q, iB, varargin )
% SETNORMALIZATION  Set normalization batch data of an 
% nlsaDiffusionOperator_batch object 
%
% Modified 2018/06/18

partition = getPartition( obj );

if ~ischar( varargin{ 1 } )
    iR = varargin{ 1 };  % varargin{ 1 } stores realization
    varargin = varargin( 2 : end );
else
    [ iB, iR ] = gl2loc( partition, iB );
end

if ~isvector( q ) || numel( q ) ~= getBatchSize( partition( iR ), iB ) 
    error( 'Incompatible number of elements in normalization vector' )
end
if isrow( q )
    q = q';
end

file = fullfile( getNormalizationPath( obj ), ... 
                 getNormalizationFile( obj, iB, iR ) );
save( file, 'q', varargin{ : } )

