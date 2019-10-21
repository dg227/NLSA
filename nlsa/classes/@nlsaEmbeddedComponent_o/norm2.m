function computeVelocity( obj, varargin )
% COMPUTEVELOCITY Compute phase space velocity (finite difference in time)
%
% Modified 2014/04/05

nS      = getNSample( obj );
nB      = getNBatch( obj );
nD      = getDimension( obj );
nE      = getEmbeddingWindow( obj );
nDE     = getEmbeddingSpaceDimension( obj );
nFD     = getFDOrder( obj );
[ nBefore, nAfter ] getNSampleFD( obj );
nSBE    = getExtraBatchSize( obj );

fdType  = getFDType( obj );
w       = getFDWeights( obj );
nW      = numel( w );
idxE    = getEmbeddingIndices( obj );
idxE    = idxE( end ) + 1 - idxE;

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Setup logfile and write calculation summary
Opt.logFile   = '';
Opt.logPath   = obj.path;
Opt.ifWriteXi = false;
Opt = parseargs( Opt, varargin{ : } );
if isempty( Opt.logFile )
    logId = 1;
else
    logId = fopen( fullfile( Opt.logPath, Opt.logFile ), 'w' );
end

clk = clock;
[ ~, hostname ] = unix( 'hostname' );
fprintf( logId, 'computeVelocity starting on %i/%i/%i %i:%i:%2.1f \n', ...
    clk( 1 ), clk( 2 ), clk( 3 ), clk( 4 ), clk( 5 ), clk( 6 ) );
fprintf( logId, 'Hostname %s \n', hostname );
fprintf( logId, 'Path %s \n', obj.path );
fprintf( logId, 'Number of samples         = %i, \n', nS );
fprintf( logId, 'Number of batches         = %i, \n', nB );
fprintf( logId, 'Physical space dimension  = %i, \n', nD );
fprintf( logId, 'Embedding window          = %i, \n', nE );
fprintf( logId, 'Embedding space dimension = %i, \n', nDE );
fprintf( logId, 'Finite-difference order   = %i, \n', nFD );
fprintf( logId, 'Finite-difference type    = %s, \n', fdType );

 
% Get samples before the first batch
if any( strcmp( fdType, { 'backward' 'central' } ) )
    tic
    xB = getData( obj, 0, 'native' );
    xB = xB( :, end - nBefore - nSBE + 1 : end - nSBE ); 
    tWall = toc;
    fprintf( logId, 'READB %2.4f \n', tWall );
else
    xB = [];
end
    
  
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Loop over the batches
for iB = 1 : nB

    nSB = getBatchSize( obj, iB );

    % Alocate data and phase space velocity arrays
    tic
    xi     = zeros( nDX, nSBE + nSB );
    tWall = toc;
    fprintf( logId, 'ALLOCATE batch %i/%i, number of samples %i %2.4f \n', ...
                iB, nB, nSB, tWall );
    
    % Get samples after the current batch
    if any( strcmp( obj.fdType, { 'central' 'forward' } ) )
        tic
        xA = getData( obj, iB + 1, 'native' );
        xA = xA( :, nSBE + 1 : nAfter + nSBE ); 
        tWall = toc;
        fprintf( logId, 'READA %2.4f \n', tWall );
    else
        xA = [];
    end 
     
    % Read current batch, concatenate with before and after samples
    tic
    xC = getData( obj, iB, native );
    x  = [ xB  xC xA ];
    tWall = toc;
    fprintf( logId, 'READ batch %i/%i %2.4f \n', iB, nB, tWall );
 
  
    % Compute finite difference
    for iW = 1 : nW
        xi = xi + w( iW ) * x( :, iW : iW + nSB + nSBE - 1 );
    end
    tWall = toc;
    fprintf( logId, 'FD batch %i/%i %2.4f \n', iB, nB, tWall );

    % Compute norm
    tic

    %xi2 = sum( xi .^ 2, 1 );
    %switch sFormat
    %    case 'evector'
    %        xiNorm = sqrt( xi2 );
    %    case 'overlap'
    %        xiNorm = zeros( 1, nSB );
    %        for iE = 1 : numel( idxE )
    %           iStart = nE + 1 - idxE( iE );
    %           xiNorm = xiNorm + xi2( iStart : iStart + nSB - 1 );
    %        end
    %        xiNorm = sqrt( xiNorm );
    %end
    tWall = toc;
    fprintf( logId, 'NORM batch %i/%i %2.4f \n', iB, nB, tWall );
    
    % Get the "before" samples for the next batch
    if any( strcmp( obj.fdType, { 'backward' 'central' } ) ) 
        xB = x( :, end - nAfter - nBefore + 1 : end - nAfter ); 
    else
        xB = [];
    end

    if Opt.ifWriteXi
        setVelocity( obj, iB, xi, xiNorm, '-v7.3' )
    else
        setVelocity( obj, iB, [], xiNorm, '-v7.3' )
    end
end

clk = clock; % Exit gracefully
fprintf( logId, 'computeVelocity finished on %i/%i/%i %i:%i:%2.1f \n', ...
    clk( 1 ), clk( 2 ), clk( 3 ), clk( 4 ), clk( 5 ), clk( 6 ) );
if ~isempty( Opt.logFile )
    fclose( logId );
end

