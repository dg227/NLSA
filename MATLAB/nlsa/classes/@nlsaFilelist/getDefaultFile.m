function file = getDefaultFile( obj, partition, prefix )
% GETDEFAULTFILE Construct filenames for an array of nlsaFilelist objects from 
% an array of nlsaPartition objects and a prefix
%
% Modified 2014/04/06


if ~isa( partition, 'nlsaPartition' )
    error( 'Second argument must be specified as an nlsaPartition object' )
end
siz = size( obj );

if numel( siz ) ~= numel( size( partition ) ) ...
  || any( siz ~= size( partition ) ) ...
  || any( getNFile( obj ) ~= getNBatch( partition ) )
    error( 'Incompatible partition' )
end

if nargin == 2
    prefix = '';
end

if ~isrowstr( prefix )
    error( 'Prefix must be a character string' )
end

prefix = { prefix, '' };

nR   = numel( obj );
file = cell( size( obj ) );
  
for iR = 1 : nR
    nB = getNBatch( partition( iR ) );
    file{ iR } = cell( 1, nB );
    if nR > 1
        prefix{ 2 } = sprintf( '_%i', iR );
    end
    for iB = 1 : nB
        idxB             = getBatchLimit( partition( iR ), iB );
        file{ iR }{ iB } = sprintf( '%s_%i-%i.mat', ...
                                    strcat( prefix{ : } ), ...
                                    idxB( 1 ), idxB( 2 ) ); 
    end
end

if nR == 1
    file = file{ 1 };
end
