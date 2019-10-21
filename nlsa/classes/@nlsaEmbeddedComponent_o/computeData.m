function computeData( obj, src, iR )
% COMPUTEDATA  Lag-embed the data in an nlsaComponent object using "overlap"
% storage format
%
% Modified 2017/07/21

if nargin == 2
    iR = 1;
end

nD   = getDimension( obj );               % physical space dimension
nDE  = getEmbeddingSpaceDimension( obj ); % embedding space dimension
nE   = getEmbeddingWindow( obj );         % max time index in lag window
idxO = getOrigin( obj );                  % embedding origin 
nXA  = getNXA( obj );                     % samples after main interval
nXB  = getNXB( obj );                     % samples before main interval

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%% VALIDATE INPUT ARGUMENTS
if ~isa( src, 'nlsaComponent' )
    error( 'Data source must be specified as an nlsaComponent object' )
end
if getDimension( src ) ~= nD
    error( 'Invalid source data dimension' )
end
if getNSample( obj ) + idxO - 1 > getNSample( src, iR )
    error( 'End index for embedding must be less than or equal to the number of source data' )
end


%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% INITIALIZATION
% Read the source batch containing the start index, taking into 
% account extra samples needed for embedding and/or nXB.
% Below, iSBSrc1 is the batch-local index in the source data
iWant   = idxO - ( nE - 1 + obj.nXB );
iBSrc   = findBatch( src, iWant, iR );
xSrc    = getData( src, iBSrc, iR );
lSrc    = getBatchLimit( src, iBSrc, iR );
nSBSrc  = getBatchSize( src, iBSrc, iR );


%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% SAMPLES BEFORE MAIN INTERVAL
iSBE1   = 1; % iSBE is the batch-local index in the embedded data
nSBE    = obj.nXB;
x       = zeros( nD, nSBE + nE - 1 );
iSBSrc1 = idxO - lSrc( 1 ) - nSBE + 1 - nE + 1;
deficit = nSBE + nE - 1;
while deficit >= 0 
    nSProvided = min( nSBSrc - iSBSrc1 + 1, nSBE + nE - iSBE1 );
    iSBSrc2    = iSBSrc1 + nSProvided - 1;
    iSBE2      = iSBE1 + nSProvided - 1;
    x( :, iSBE1 : iSBE2 ) = xSrc( :, iSBSrc1 : iSBSrc2 );
    iSBE1   = iSBE2 + 1;
    iSBSrc1 = iSBSrc2 + 1;        
    deficit = nSBE + nE - 1 - iSBE1;
    if deficit >= 0 && iSBSrc1 > nSBSrc
        iBSrc   = iBSrc + 1;
        iKeep1  = nSBSrc - obj.idxE( end ) + 2;
        iKeep2  = nSBSrc; 
        xKeep   = xSrc( :, iKeep1 : iKeep2 );
        xSrc    = [ xKeep getData( src, iBSrc, iR ) ];
        nSKeep  = size( xKeep, 2 );
        iSBSrc1 = 1 + nSKeep; 
        nSBSrc  = getBatchSize( src, iBSrc, iR ) + nSKeep;
    end
end
xOverlap = x( :, end - nE + 2 : end );
if obj.nXB > 0
    setData_before( obj, x, '-v7.3' )
end  

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%% MAIN-INTERVAL SAMPLES
% Loop over the embedded data batches
for iBE = 1 : getNBatch( obj )
    nSBE               = getBatchSize( obj, iBE );
    x                  = zeros( nD, nSBE + nE - 1 );
    x( :, 1 : nE - 1 ) = xOverlap;
    iSBE1              = nE; % iSBE is the batch-local index in the embedded data
    deficit            = nSBE;
    while deficit >= 0 
        nSProvided = min( nSBSrc - iSBSrc1 + 1, nSBE + nE - iSBE1 );
        iSBSrc2    = iSBSrc1 + nSProvided - 1;
        iSBE2      = iSBE1 + nSProvided - 1;
        x( :, iSBE1 : iSBE2 ) = xSrc( :, iSBSrc1 : iSBSrc2 );
        iSBE1   = iSBE2 + 1;
        iSBSrc1 = iSBSrc2 + 1;        
        deficit = nSBE + nE - 1 - iSBE1;
        if deficit >= 0 && iSBSrc1 > nSBSrc
            iBSrc   = iBSrc + 1;
            iKeep1  = nSBSrc - obj.idxE( end ) + 2;
            iKeep2  = nSBSrc; 
            xKeep   = xSrc( :, iKeep1 : iKeep2 );
            xSrc    = [ xKeep getData( src, iBSrc, iR ) ];
            nSKeep  = size( xKeep, 2 );
            iSBSrc1 = 1 + nSKeep; 
            nSBSrc  = getBatchSize( src, iBSrc, iR ) + nSKeep;
        end
    end
    setData( obj, x, iBE, '-v7.3' )
    xOverlap =  x( :, end - nE + 2 : end );
end    

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% SAMPLES AFTER MAIN INTERVAL
if obj.nXA > 0
    nSBE  = obj.nXA;
    x     = zeros( nD, nSBE + nE - 1 );
    x( :, 1 : nE - 1 ) = xOverlap;
    iSBE1   = nE;
    deficit = nSBE;
    while deficit >= 0 
        nSProvided = min( nSBSrc - iSBSrc1 + 1, nSBE + nE - iSBE1 );
        iSBSrc2    = iSBSrc1 + nSProvided - 1;
        iSBE2      = iSBE1 + nSProvided - 1;
        x( :, iSBE1 : iSBE2 ) = xSrc( :,  iSBSrc1 : iSBSrc2 );
        iSBE1   = iSBE2 + 1;
        iSBSrc1 = iSBSrc2 + 1;        
        deficit = nSBE + nE - 1 - iSBE1;
        if deficit >= 0 && iSBSrc1 > nSBSrc
            iBSrc   = iBSrc + 1;
            iKeep1  = nSBSrc - obj.idxE( end ) + 2;
            iKeep2  = nSBSrc; 
            xKeep   = xSrc( :, iKeep1 : iKeep2 );
            xSrc    = [ xKeep getData( src, iBSrc, iR ) ];
            nSKeep  = size( xKeep, 2 );
            iSBSrc1 = 1 + nSKeep; 
            nSBSrc  = getBatchSize( src, iBSrc, iR ) + nSKeep;
        end
    end
    setData_after( obj, x, '-v7.3' )
end  
