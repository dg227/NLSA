function cv = computeRightCovariance( obj, src, varargin )
% COMPUTERIGHTCOVARIANCE Compute right (temporal) covariance matrix from 
% time-lagged embedded data src 
% 
% Modified 2016/02/02


%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Do some basic checking of input arguments
if ~isscalar( obj )
    error( 'First argument must be scalar' )
end
if ~isa( src, 'nlsaEmbeddedComponent' )
    error( 'Embedded data must be specified as an array of nlsaEmbeddedComponent objects' )
end
if ~isCompatible( obj, src )
    error( 'Incompatible embedded data' )
end

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Setup a global partition for the embedded data
% idxBG are the global indices of the batch boundaries
partition = getPartition( obj );
[ partitionG, idxBG ] = mergePartitions( partition );
partitionD = getSpatialPartition( obj );
nDE    = getBatchSize( partitionD );
nC     = numel( nDE );
nR     = numel( partition );
nSB    = getBatchSize( partitionG );
nSTot  = getNSample( partitionG );
nBG    = getNBatch( partitionG );
nDETot = sum( nDE );

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Parse optional input arguments
Opt.logFile            = '';
Opt.logPath            = obj.path;
Opt.logFilePermissions = 'w';
Opt.ifWriteOperator    = true;
Opt = parseargs( Opt, varargin{ : } );

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Setup logfile and write calculation summary
if isempty( Opt.logFile )
    logId = 1;
else
    logId = fopen( fullfile( Opt.logPath, Opt.logFile ), Opt.logFilePermissions );
end

clk = clock;
[ status hostname ] = unix( 'hostname' );
fprintf( logId, 'computeRightCovariance starting on %i/%i/%i %i:%i:%2.1f \n', ...
    clk( 1 ), clk( 2 ), clk( 3 ), clk( 4 ), clk( 5 ), clk( 6 ) );
fprintf( logId, 'Hostname %s \n', hostname );
fprintf( logId, 'Path %s \n', obj.path );
fprintf( logId, 'Total dimension                         = %i \n', sum( nDETot ) );
fprintf( logId, 'Number of samples                       = %i \n', nSTot );
fprintf( logId, 'Number of realizations                  = %i \n', nR );
fprintf( logId, 'Number of components                    = %i \n', nC );
fprintf( logId, 'Number of batches                       = %i \n', nBG );
fprintf( logId, 'Max batch size                          = %i \n', max( nSB ) );
fprintf( logId, 'Min batch size                          = %i \n', min( nSB ) );
fprintf( logId, 'Max compoment embedding space dimension = %i \n', max( nDE ) );
fprintf( logId, 'Min component embedding space dimension = %i \n', min( nDE ) );
fprintf( logId, '----------------------------------------- \n' ); 


%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Form right covariance
cv = zeros( nSTot );

% Loop over the source components
for iC = 1 : nC

    iS1 = 1;

    % Loop over the upper triangular blocks
    for iBG = 1 : nBG

        iR   = idxBG( 1, iBG );
        iB   = idxBG( 2, iBG );
        nBI  = getNBatch( partition( iR ) );
        nSBI = getBatchSize( partitionG, iBG );

        iS2 = iS1 + nSBI - 1;

        tic
        xI    = getData( src( iC, iR ), iB ); 
        tWall = toc;
        fprintf( logId, 'READXI component %i/%i, realization %i/%i, local batch %i/%i, global batch %i/%i %2.4f \n', ...
            iC, nC, iR, nR, iB, nBI, iBG, nBG, tWall );

        tic
        cv( iS1 : iS2, iS1 : iS2 ) = cv( iS1 : iS2, iS1 : iS2 ) + xI' * xI;
        tWall = toc;
        fprintf( logId, 'CV component %i/%i, realization %i-%i, local batches %i-%i, global batches %i-%i %2.4f \n', ...
            iC, nC, iR, iR, iB, iB, iBG, iBG, tWall );

        jS1 = iS2 + 1;
        for jBG = iBG + 1 : nBG

            jR   = idxBG( 1, jBG );
            jB   = idxBG( 2, jBG );
            nBJ  = getNBatch( partition( jR ) );
            nSBJ = getBatchSize( partitionG, jBG );
 
            jS2 = jS1 + nSBJ - 1;

            tic
            xJ = getData( src( iC, jR ), jB );
            tWall = toc;
            fprintf( logId, 'READXJ component %i/%i, realization %i/%i, local batch %i/%i, global batch %i/%i %2.4f \n', ...
                iC, nC, jR, nR, jB, nBJ, jBG, nBG, tWall );

            tic
            cv( iS1 : iS2, jS1 : jS2 ) = cv( iS1 : iS2, jS1 : jS2 ) + xI' * xJ;
            tWall = toc;
            fprintf( logId, 'CV component %i/%i, realization %i-%i, local batches %i-%i, global batches %i-%i %2.4f \n', ...
                iC, nC, iR, jR, iB, jB, iBG, jBG, tWall );

            jS1 = jS2 + 1;
        end % J batch loop

        iS1 = iS2 + 1;

    end % I batch loop
end % component loop 

% Fill in the lower triangular blocks
jS1 = 1;
for jBG = 1 : nBG

    jR   = idxBG( 1, jBG );
    jB   = idxBG( 2, jBG );
    nBJ  = getNBatch( partition( jR ) );
    nSBJ = getBatchSize( partitionG, jBG );

    jS2 = jS1 + nSBJ - 1;

    iS1 = jS2 + 1;
    for iBG = jBG + 1 : nBG

        iR   = idxBG( 1, iBG );
        iB   = idxBG( 2, iBG );
        nBI  = getNBatch( partition( iR ) );
        nSBI = getBatchSize( partitionG, iBG );
 
        iS2 = iS1 + nSBI - 1;

        tic
        cv( iS1 : iS2, jS1 : jS2 ) = cv( jS1 : jS2, iS1 : iS2 )';
        tWall = toc;
        fprintf( logId, 'SYM component %i/%i, realization %i-%i, local batches %i-%i, global batches %i-%i %2.4f \n', ...
                iC, nC, iR, jR, iB, jB, iBG, jBG, tWall );

        iS1 = iS2 + 1;
    end % I batch loop

    jS1 = jS2 + 1;

end % J batch loop


if Opt.ifWriteOperator
    tic
    setRightCovariance( obj, cv, '-v7.3' )
    tWall = toc;
    fprintf( logId, 'WRITECV %2.4f \n', tWall );
end

clk = clock; % Exit gracefully
fprintf( logId, 'computeRightCovariance finished on %i/%i/%i %i:%i:%2.1f \n', ...
    clk( 1 ), clk( 2 ), clk( 3 ), clk( 4 ), clk( 5 ), clk( 6 ) );
if ~isempty( Opt.logFile )
    fclose( logId );
end
