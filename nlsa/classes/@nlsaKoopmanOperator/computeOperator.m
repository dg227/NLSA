function V = computeOperator( obj, diffOp, varargin )
% COMPUTEOPERATOR Compute Koopman generator in an eigenbasis of a kernel
% integral operator.
% 
% Modified 2020/04/15

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Validate input arguments and determine array sizes
if ~isa( diffOp, 'nlsaKernelOperator' ) || ~isscalar( diffOp )
    error( 'Third input argument must be a scalar nlsaKernelOperator object' )
end
partition  = getPartition( obj ); % partition for in-sample data
nR  = numel( partition );      % number of realization
if nR ~= numel( getPartition( diffOp ) ) ...
   ||  any( ~isequal( partition, getPartition( diffOp ) ) )
    error( 'Incompatible partition of kernel operator' )
end
[ partitionG, idxG ] = mergePartitions( partition ); % global partition
nSR = getNSample( partition ); % number of samples in each realization
nS  = sum( nSR );              % total number of samples
nT  = nS / nR;                 % number of temporal samples
if any( nSR ~= nT )   
    error( 'Non-equal number of samples per realization not supported.' )
end
idxPhi = getBasisFunctionIndices( obj ); 
if idxPhi( end ) > getNEigenfunction( diffOp )
    msgStr = [ 'Requested basis functions exceed those available from ' ...
               'the kernel integral operator.' ]; 
    error( msgStr ) 
end
nPhi         = numel( idxPhi );     % number of diffusion eigenfunctions 
nFD          = getFDOrder( obj );   % finite-difference order
[ nTB, nTA ] = getNSampleFD( obj ); % temporal samples before (nTB) and
                                    % and after (nTA) main finite-difference
                                    % interval
nTFD         = nT - nTB - nTA;      % temporal samples after finite difference
nSFD         = nTFD * nR;           % total samples after finite difference
fdT          = getFDType( obj );    % finite difference type
ifAntisym    = getAntisym( obj );   % enforce antisymmetric metric

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
    logId = fopen( fullfile( Opt.logPath, Opt.logFile ), ...
                   Opt.logFilePermissions );
end

clk = clock;
[ ~, hostname ] = unix( 'hostname' );
fprintf( logId, 'computeOperator starting on %i/%i/%i %i:%i:%2.1f \n', ...
    clk( 1 ), clk( 2 ), clk( 3 ), clk( 4 ), clk( 5 ), clk( 6 ) );
fprintf( logId, 'Hostname %s \n', hostname );
fprintf( logId, 'Path %s \n', obj.path );
fprintf( logId, 'Number of samples               = %i, \n', nS );
fprintf( logId, 'Number of realizations          = %i, \n', nR );
fprintf( logId, 'Number of temporal samples      = %i, \n', nT );
fprintf( logId, 'Finite-difference order         = %i, \n', nFD );
fprintf( logId, 'Finite-difference type          = %s, \n', fdT );
fprintf( logId, 'Antisymmetrization              = %i, \n', ifAntisym );
fprintf( logId, 'Number of basis functions       = %i, \n', nPhi );
fprintf( logId, 'Basis function incices          = %s \n', idx2str( idxPhi ) );

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Read the basis functions and inner product weights
tWall0 = tic;
[ phi, mu ] = getEigenfunctions( diffOp );
phi = phi( :, idxPhi );
tWall = toc( tWall0 );
fprintf( logId, 'READPHI %2.4f \n', tWall );


%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Perform temporal finite difference 
tWall0 = tic;
phi  = reshape( phi, [ nT nR nPhi ] );
dPhi = reshape( shiftdim( phi, 1 ), [ nR * nPhi, nT ] ); 
dPhi = computeFD( obj, dPhi ); 
dPhi = shiftdim( reshape( dPhi, [ nR nPhi nTFD ] ), - 1 );
dPhi = reshape( dPhi, [ nSFD nPhi ] );
tWall = toc( tWall0 );
fprintf( logId, 'FD %2.4f \n', tWall );

% Form operator matrix elements
tWall0 = tic;
mu    = reshape( mu, [ nT nR ] );
phiMu = phi( nTB + 1 : end - nTA, :, : ) .* mu( nTB + 1 : end - nTA, : );
phiMu = reshape( phiMu, [ nSFD nPhi ] );
V     = phiMu' * dPhi; 
tWall = toc( tWall0 );
fprintf( logId, 'GEN %2.4f \n', tWall );

% Antisymmetrize if requested
if ifAntisym
    tWall0 = tic;
    V = ( V - V' ) / 2;
    tWall = toc( tWall0 );
    fprint( logId, 'ANTISYM %2.4f \n', tWall );
end

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
if Opt.ifWriteOperator
    tWall0 = tic;
    setOperator( obj, V, '-v7.3' )
    tWall = toc( tWall0 );
    fprintf( logId, 'WRITEV %2.4f \n', tWall ); 
end

clk = clock; % Exit gracefully
fprintf( logId, 'computeOperator finished on %i/%i/%i %i:%i:%2.1f \n', ...
    clk( 1 ), clk( 2 ), clk( 3 ), clk( 4 ), clk( 5 ), clk( 6 ) );
if ~isempty( Opt.logFile )
    fclose( logId );
end
