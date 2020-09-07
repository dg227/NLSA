function computeData( obj, srcX, srcT, iCRec, iRRec, varargin )
% COMPUTEDATA Perform reconstruction in physical space from spatial and
% temporal patterns
%
% Modified 2020/09/04

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Validate input arguments
msgId = [ obj.getErrMsgId ':computeData:' ];
if ~( isa( srcX, 'nlsaProjectedComponent' ) || ...
    isa( srcX, 'nlsaLinearMap' ) )
    msgStr = 'Second argument must be an nlsaProjectedComponent or nlsaLinearMap object.'; 
    error( [ msgId 'invalidSrcX' ], msgStr ) 
end
if ~isa( srcT, 'nlsaKernelOperator' )
    msgStr = 'Third input argument must be an nlsaKernelOperator object.'; 
    error( [ msgId 'invalidSrcT' ], msgStr )
end
           
nD  = getDimension( obj );
nDE = getEmbeddingSpaceDimension( srcX, iCRec );
nE  = nDE / nD;
if  ~isposint( nE )
    error( 'Incompatible dimensions' )
end

partition  = getPartition( obj );
partitionE = getPartition( srcT );
nS = getNSample( partition );
nB  = getNBatch( partition );
nSE = getNSample( partitionE( iRRec ) );
nSBE = getBatchSize( partitionE( iRRec ) );
if nS ~= nSE + nE - 1
    error( 'Incompatible number of samples' )
end
idxPhi = getBasisFunctionIndices( obj );
%idxPhi = idxPhi( idxPhi <= nDE );
u = getSpatialPatterns( srcX, iCRec );


%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Parse optional input arguments
Opt.batch   = 1 : getNBatch( partition );
Opt.logFile = '';
Opt.logPath = getDataPath( obj );
Opt         = parseargs( Opt, varargin{ : } );

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Setup logfile, write calculation summary
if isempty( Opt.logFile )
    logId = 1;
else
    logId = fopen( fullfile( Opt.logPath, Opt.logFile ), 'w' );
end
clk = clock;
[ ~, hostname ] = unix( 'hostname' );
fprintf( logId, 'computeData starting on %i/%i/%i %i:%i:%2.1f \n', ...
    clk( 1 ), clk( 2 ), clk( 3 ), clk( 4 ), clk( 5 ), clk( 6 ) );
fprintf( logId, 'Hostname %s \n', hostname );
fprintf( logId, 'Path %s \n', obj.path );

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% RECONSTRUCTION
% Loop over the out-of-sample realizations and batches
for iBRec = 1 : numel( Opt.batch )
    iHave = [ -Inf -Inf ];
    iB = Opt.batch( iBRec );
    lB = getBatchLimit( partition, iB );
    nSB = getBatchSize( partition, iB );
    x = zeros( nD, nSB );
    iWant( 1 ) = max( lB( 1 ) - nE + 1, 1 );
    iWant( 2 ) = min( lB( 2 ), nSE );
    nSERec = iWant( 2 ) - iWant( 1 ) + 1;
    xE = zeros( nDE, nSERec );
    deficit = nSERec;
    iSE1 = 1;
    while deficit > 0
        nSProvided = iHave( 2 ) - iWant( 1 ) + 1;
        if nSProvided <= 0
            iBE = findBatch( partitionE( iRRec ), iWant( 1 ) );
            v    = getTemporalPatterns( srcT, iBE, iRRec );
            iHave = getBatchLimit( partitionE( iRRec ), iBE );
            iV1 = iWant( 1 ) - iHave( 1 ) + 1;
        else
            iSE2 = iSE1 + nSProvided - 1;
            iSE2 = min( iSE2, nSERec );
            nSAdd = iSE2 - iSE1 + 1;
            iV2  = iV1 + nSAdd - 1;
            xE( :, iSE1 : iSE2 ) = u( :, idxPhi ) * v( iV1 : iV2, idxPhi ).';
            deficit = deficit - nSAdd;
            iWant( 1 ) = iWant( 1 ) + nSAdd;
            iSE1 = iSE2 + 1;
        end
    end
    iSEO = min( lB( 1 ), nE ); % origin column in delay-space data array xE 
    for iSB = 1 : nSB
        iSE = iSEO + iSB - 1;  % current column in xE
        jE = iSE - nE + 1 : iSE; 
        iJE1 = find( jE > 0, 1, 'first' );
        iJE2 = find( jE <= nSERec, 1, 'last' );
        % jE and iE are the column and row indices, respectively, in xE that 
        % will be summed over. idxE are the corresponding linear array indices
        jE = repmat( jE( iJE1 : iJE2 ), [ nD 1 ] ); 
        iE = ( 0 : nD - 1 )' + ( nDE - nD + 1 : -nD : 1 );
        iE = iE( :,  iJE1 : iJE2 );
        idxE = sub2ind( [ nDE nSERec ], iE, jE );
        x( :, iSB ) = mean( xE( idxE ), 2 );
    end

    tic 
    setData( obj, x, iB, '-v7.3' )
    tWall = toc;
end % global batch loop

clk = clock; % Exit gracefully
fprintf( logId, 'computeData finished on %i/%i/%i %i:%i:%2.1f \n', ...
    clk( 1 ), clk( 2 ), clk( 3 ), clk( 4 ), clk( 5 ), clk( 6 ) );
if ~isempty( Opt.logFile )
    fclose( logId );
end
