function [ model, In ] = clausNLSAModel_koopman( experiment, regMethod, epsilon, idxPhi, idxPsi, idxPsiRec )
%% CLAUSNLSAMODEL_KOOPMAN Build NLSA model for CLAUS data with Koopman 
%  eigenfunctions. This function is to be used in conjunction with the scripts
%  clausKoopmanEigenfunctions and diff2Koop. 
% 
% regMethod:  Diffusion regularization method
% epsilon:    Diffusion regularization parameter
% idxPhi:     NLSA eigenfunctions used for Koopman eigenvalue problemn
% idxPsi:     Koopman eigenfunctions to store in Koopman NLSA model
% idxPSiRec:  Koopman eigenfunctions to reconstruct 

if nargin == 0
    experiment = 'lo_res';
end

% Get oringal NLSA model
[ model_orig, In ] = clausNLSAModel_den( experiment );

%% NLSA MODEL FOR KOOPMAN
In.kEpsilon = epsilon;
In.kIdxPsi = idxPsi; 
In.idxPhiRec  = idxPsiRec + 1;    % eigenfunctions for reconstruction
In.nPhi       = 2 * numel( In.kIdxPsi ) + 1;
In.nPhiPrj    = In.nPhi;
strPhi = idx2str( idxPhi, 'idxPhi' );
strPsi = idx2str( In.kIdxPsi, 'idxPsi' );
strK = [ sprintf( '%s_epsilon%1.3g_', regMethod, In.kEpsilon ) strPhi '_' strPsi ];
nlsaPath = fullfile( './data/koopman/', strK );    % path for NLSA code output

%==============================================================================
% Determine total number of samples, time origin for delay embedding

tNum    = datenum( In.tLim{ 1 }, In.tFormat ) ...
        : 1/8 * In.nDT ... 
        : datenum( In.tLim{ 2 }, In.tFormat ); % timestamps
In.nS      = numel( tNum );                    % sample number
In.nC      = numel( In.Src );  % number of source components
In.nCTrg   = numel( In.Trg );  % number of target compoents
In.Src( 1 ).idxE = 1 : In.Src( 1 ).nE;
In.nE = In.Src( 1 ).nE;
for iC = 2 : In.nC
    In.Src( iC ).idxE = 1 : In.Src( iC ).nE;
    In.nE = max( In.nE, In.Src( iC ).nE );
end
In.Trg( 1 ).idxE  = 1 : In.Trg( 1 ).nE;
In.nETrg = In.Trg( 1 ).nE;
for iC = 2 : In.nCTrg
    In.Trg( iC ).idxE  = 1 : In.Trg( iC ).nE;
    In.nETrg = max( In.nETrg, In.Trg( iC ).nE );
end
In.idxT1   = max( In.nE, In.nETrg ) + In.nXB;    % time origin for embedding
In.nSE     = In.nS - In.idxT1 + 1 - In.nXA;      % sample number after embedding

%==============================================================================
% Determine data dimensions
for iC = 1 : In.nC
    In.Src( iC ).nD = ...
    numel( In.Src( iC ).xLim( 1 ) : In.Src( iC ).nDX / 2 : In.Src( iC ).xLim( 2 ) ); 
    if strcmp( In.Src( iC ).field, 'tb' )
        In.Src( iC ).nD = In.Src( iC ).nD ...
        * numel( In.Src( iC ).yLim( 1 ) : In.Src( iC ).nDY / 2 : In.Src( iC ).yLim( 2 ) );
    end
end 
for iC = 1 : In.nCTrg
    In.Trg( iC ).nD = ...
    numel( In.Trg( iC ).xLim( 1 ) : In.Trg( iC ).nDX / 2 : In.Trg( iC ).xLim( 2 ) ); 
    if strcmp( In.Trg( iC ).field, 'tb' )
        In.Trg( iC ).nD = In.Trg( iC ).nD ...
        * numel( In.Trg( iC ).yLim( 1 ) : In.Trg( iC ).nDY / 2 : In.Trg( iC ).yLim( 2 ) );
    end
end 
        
%==============================================================================
% Setup nlsaComponent and nlsaProjectedComponent objects 
partition = nlsaPartition( 'nSample', In.nS );

for iC = In.nC : -1 : 1

    str = [ 'x'    int2str( In.Src( iC ).xLim( 1 ) ) '-' int2str( In.Src( iC ).xLim( 2 ) ) ...
            '_y'   int2str( In.Src( iC ).yLim( 1 ) ) '-' int2str( In.Src( iC ).yLim( 2 ) ) ...
            '_'    In.tLim{ 1 }, '-', In.tLim{ 2 } ...
            '_nDX' int2str( In.Src( iC ).nDX ) ...
            '_nDY' int2str( In.Src( iC ).nDY ) ...
            '_nDT' int2str( In.nDT ) ];

    path = fullfile( pwd, 'data/raw', In.Src( iC ).field, str );
    fList = nlsaFilelist( 'file', 'dataAnomaly.mat' );
    tagC  = [ In.Src( iC ).field ... 
              '_x'    int2str( In.Src( iC ).xLim( 1 ) ) '-' int2str( In.Src( iC ).xLim( 2 ) ) ...
            '_y'   int2str( In.Src( iC ).yLim( 1 ) ) '-' int2str( In.Src( iC ).yLim( 2 ) ) ...
            '_nDX' int2str( In.Src( iC ).nDX ) ...
            '_nDY' int2str( In.Src( iC ).nDY ) ]; 

    tagR = [ In.tLim{ 1 } '-' In.tLim{ 2 } ...
             '_nDT' int2str( In.nDT ) ];

    srcComponent( iC ) = nlsaComponent( 'partition',    partition, ...
                                        'dimension',    In.Src( iC ).nD, ...
                                        'path',         path, ...
                                        'file',         fList, ...
                                        'componentTag',   tagC, ...
                                        'realizationTag', tagR );

    if In.nXB == 0 && In.nXA == 0
        embComponent( iC ) = nlsaEmbeddedComponent_o( ...
                                 'idxE', In.Src( iC ).idxE, ...
                                 'nXB', In.nXB, ...
                                 'nXA', In.nXA );

    else
        embComponent( iC ) = nlsaEmbeddedComponent_xi_o( ...
                                 'idxE', In.Src( iC ).idxE, ...
                                 'nXB', In.nXB, ...
                                 'nXA', In.nXA, ...
                                 'fdOrder',  In.fdOrder, ...
                                 'fdType', In.fdType );
    end

end


for iC = In.nCTrg : -1 : 1

    str = [ 'x',    int2str( In.Trg( iC ).xLim( 1 ) ), '-', int2str( In.Trg( iC ).xLim( 2 ) ), ...
            '_y',   int2str( In.Trg( iC ).yLim( 1 ) ), '-', int2str( In.Trg( iC ).yLim( 2 ) ), ...
            '_',    In.tLim{ 1 }, '-', In.tLim{ 2 }, ...
            '_nDX', int2str( In.Trg( iC ).nDX ), ...
            '_nDY', int2str( In.Trg( iC ).nDY ), ...
            '_nDT', int2str( In.nDT ) ];

    path = fullfile( pwd, 'data/raw', In.Trg( iC ).field, str );
    fList = nlsaFilelist( 'file', 'dataAnomaly.mat' );

    tagC  = [ In.Trg( iC ).field ... 
              '_x' int2str( In.Trg( iC ).xLim( 1 ) ) '-' int2str( In.Trg( iC ).xLim( 2 ) ) ...
            '_y' int2str( In.Trg( iC ).yLim( 1 ) ) '-' int2str( In.Trg( iC ).yLim( 2 ) ) ...
            '_nDX' int2str( In.Trg( iC ).nDX ) ...
            '_nDY' int2str( In.Trg( iC ).nDY ) ]; 

    tagR = [ In.tLim{ 1 } '-' In.tLim{ 2 } ...
             '_nDT' int2str( In.nDT ) ];

    trgComponent( iC ) = nlsaComponent( 'partition',    partition, ...
                                        'dimension',    In.Trg( iC ).nD, ...
                                        'path',         path, ...
                                        'file',         fList, ...
                                        'componentTag', tagC, ...
                                        'realizationTag', tagR );

    if In.nXB == 0 && In.nXA == 0 
        trgEmbComponent( iC ) = nlsaEmbeddedComponent_o( ...
                  'idxE', In.Trg( iC ).idxE, ...
                  'nXB', In.nXB, ...
                  'nXA', In.nXA );
    else
            trgEmbComponent( iC ) = nlsaEmbeddedComponent_xi_o( ...
                  'idxE', In.Trg( iC ).idxE, ...
                  'nXB', In.nXB, ...
                  'nXA', In.nXA, ...
                  'fdOrder',  In.fdOrder, ...
                  'fdType', In.fdType );
    end

    prjComponent( iC ) = nlsaProjectedComponent_xi( ...
                           'nBasisFunction', In.nPhiPrj );

end

embPartition    = nlsaPartition( 'nSample', In.nSE, 'nBatch', In.nB );
trgEmbPartition = nlsaPartition( 'nSample', In.nSE, 'nBatch', In.nBTrg );

%=================================================================================
% Pairwise distance for density estimation
modeStr = 'implicit';
switch In.denLDist
    case 'l2' % L^2 distance
        denLDist = nlsaLocalDistance_l2( 'mode', modeStr );

    case 'at' % "autotuning" NLSA kernel
        denLDist = nlsaLocalDistance_at( 'mode', modeStr );

    case 'cone' % cone kernel
        denLDist = nlsaLocalDistance_cone( 'mode', modeStr, ...
                                           'normalization', In.denDistNorm, ...
                                           'zeta', In.denZeta, ...
                                           'tolerance', In.tol, ...
                                           'alpha', In.denAlpha );
end

denDFunc = nlsaLocalDistanceFunction( 'localDistance', denLDist );

denPDist = nlsaPairwiseDistance( 'nearestNeighbors', In.nN, ...
                                 'distanceFunction', denDFunc );

%================================================================================
%Kernel density estimation

switch In.denType
    case 'fb' % fixed bandwidth
        den = nlsaKernelDensity_fb( ...
                 'dimension',              In.denND, ...
                 'bandwidthBase',          In.denEpsilonB, ...
                 'bandwidthExponentLimit', In.denEpsilonE, ...
                 'nBandwidth',             In.denNEpsilon );

    case 'vb' % variable bandwidth 
        den = nlsaKernelDensity_vb( ...
                 'dimension',              In.denND, ...
                 'kNN',                    In.denNN, ...
                 'bandwidthBase',          In.denEpsilonB, ...
                 'bandwidthExponentLimit', In.denEpsilonE, ...
                 'nBandwidth',             In.denNEpsilon );
end

%==============================================================================
% Pairwise distances for in-sample data
switch In.lDist
    case 'l2' % L^2 distance
        lDist = nlsaLocalDistance_l2( 'mode', modeStr );

    case 'at' % "autotuning" NLSA kernel
        lDist = nlsaLocalDistance_at( 'mode', modeStr );

    case 'cone' % cone kernel
        lDist = nlsaLocalDistance_cone( 'mode', modeStr, ...
                                        'zeta', In.zeta, ...
                                        'tolerance', In.tol, ...
                                        'alpha', In.coneAlpha );
end
lScl  = nlsaLocalScaling_pwr( 'pwr', 1 / In.denND );
dFunc = nlsaLocalDistanceFunction_scl( 'localDistance', lDist, ...
                                       'localScaling', lScl );
pDist = nlsaPairwiseDistance( 'distanceFunction', dFunc, ...
                              'nearestNeighbors', In.nN );


%==============================================================================
% Symmetrized pairwise distances
sDist = nlsaSymmetricDistance_gl( 'nearestNeighbors', In.nNS );

%==============================================================================
% Diffusion operators 

switch In.diffOpType
    % global storage format, fixed bandwidth
    case 'gl'
        diffOp = nlsaDiffusionOperator_gl(    'alpha',          In.alpha, ...
                                              'epsilon',        In.epsilon, ...
                                              'nEigenfunction', In.nPhi );

    % global storage format, multiple bandwidth (automatic bandwidth selection)
    case 'gl_mb'
        diffOp = nlsaDiffusionOperator_gl_mb( ...
                     'alpha',                  In.alpha, ...
                     'epsilon',                In.epsilon, ...
                     'nEigenfunction',         In.nPhi, ...
                     'bandwidthBase',          In.epsilonB, ...
                     'bandwidthExponentLimit', In.epsilonE, ...
                     'nBandwidth',             In.nEpsilon );
end

%==============================================================================
% Linear map for SVD of the target data 
linMap = nlsaLinearMap_gl( 'basisFunctionIdx', In.idxPhiSVD );

%==============================================================================
% Reconstructed components
In.nSRec  = In.nSE + In.nE - 1; % number of reconstructed samples

% Partition for reconstructed data
recPartition = nlsaPartition( 'nSample', In.nSRec, ... 
                              'nBatch',  In.nBRec );

% Reconstructed data from diffusion eigenfnunctions
recComponent = nlsaComponent_rec_phi( 'partition', recPartition, ...
                                      'basisFunctionIdx', In.idxPhiRec );

% Reconstructed data from SVD 
svdRecComponent = nlsaComponent_rec_phi( 'partition', recPartition, ...
                                         'basisFunctionIdx', In.idxVTRec );

%==============================================================================
% Build NLSA model    
model = nlsaModel_den( 'path',                            nlsaPath, ...
                       'timeFormat',                      In.tFormat, ...
                       'srcTime',                         tNum, ...
                       'sourceComponent',                 srcComponent, ...
                       'embeddingOrigin',                 In.idxT1, ...
                       'embeddingTemplate',               embComponent, ...
                       'embeddingPartition',              embPartition, ...
                       'denPairwiseDistanceTemplate',     denPDist, ...
                       'kernelDensityTemplate',           den, ...
                       'pairwiseDistanceTemplate',        pDist, ...
                       'symmetricDistanceTemplate',       sDist, ...
                       'diffusionOperatorTemplate',       diffOp, ...
                       'targetComponent',                 trgComponent, ...
                       'targetEmbeddingTemplate',         trgEmbComponent, ...
                       'projectionTemplate',              prjComponent, ...
                       'reconstructionTemplate',          recComponent, ...
                       'linearMapTemplate',               linMap, ...
                       'svdReconstructionTemplate',       svdRecComponent );


%==============================================================================
% Assign embedded components to those from original NLSA model 
model.embComponent    = model_orig.embComponent;
model.trgEmbComponent = model_orig.trgEmbComponent;
