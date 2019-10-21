function [ model, In ] = clausNLSAModel( experiment )
%% CLAUSNLSAMODEL_DEN Build NLSA model for the CLAUS brightness 
%  temperature dataset. 
% 
%  In is a data structure containing the model parameters (named after 
%  "in-sample," as opposed to "out-of-sample" data).
%
%  The data must be imported in a format compatible with the NLSA code
%  using the scripts clausImport and clausImport_av for 2D and equatorially
%  averaged data, respectively. 
%
%  For additional information on the arguments of nlsaModel( ... ) see 
%
%      ../classes/nlsaModel_base/parseTemplates.m
%      ../classes/nlsaModel/parseTemplates.m
%
% Modified 2017/11/10
 

if nargin == 0
    experiment = 'belt15';
end


switch experiment

    % Equatorial belt +/- 15o 
    case 'belt15'
        % In-sample dataset parameters
%        In.tLim           = { '1983070100' '2009063021' }; % time limits 
%        In.tLim           = { '1984010100' '2006063021' }; % time limits 
        In.tLim           = { '1984010100' '1984033021' }; % time limits 
        In.tFormat        = 'yyyymmddhh';                  % time format
        In.nDT            = 2;                             % time downsampling
        In.Src( 1 ).field = 'tb';                          % source data 
        In.Src( 1 ).xLim  = [ 0 359.5  ];                  % longitude limits
        In.Src( 1 ).yLim  = [ -15 15   ];                  % lattitude limits
        In.Src( 1 ).nDX   = 2;                             % long. downsampling
        In.Src( 1 ).nDY   = 2;                             % lat. downsampling
        In.Src( 1 ).nE    = 12;                           % embedding window length
        In.Trg            = In.Src;

        % NLSA parameters
        In.nXB          = 1;          % samples to leave out before main interval 
        In.nXA          = 1;          % samples to leave out after main interval
        In.fdOrder      = 1;          % finite-difference order 
        In.fdType       = 'backward'; % finite-difference type
        In.nB           = 32;         % batches to partition the source data
        In.nBTrg        = In.nB;      % batches to partition the target data
        In.nBRec        = In.nB;      % batches for reconstructed data
        In.nN           = 10; %5000;       % nearest neighbors for pairwise distances
        In.lDist        = 'at';       % local distance
        In.tol          = 0;          % 0 distance threshold (for cone kernel)
        In.zeta         = 0.995;      % cone kernel parameter 
        In.coneAlpha    = 1;          % velocity exponent in cone kernel
        In.nNS          = 10; %In.nN;      % nearest neighbors for symmetric distance
        In.diffOpType   = 'gl_mb';    % diffusion operator type
        In.epsilon      = 2;          % kernel bandwidth parameter 
        In.epsilonB     = 2;          % kernel bandwidth base
        In.epsilonE     = [ -40 40 ]; % kernel bandwidth exponents 
        In.nEpsilon     = 100;        % number of exponents for bandwidth tuning
        In.alpha        = 1;          % diffusion maps normalization 
        In.nPhi         = 101;        % diffusion eigenfunctions to compute
        In.nPhiPrj      = In.nPhi;    % eigenfunctions to project the data
        In.idxPhiRec    = 1 : 5;      % eigenfunctions for reconstruction
        In.idxPhiSVD    = 1 : 15;     % eigenfunctions for linear mapping
        In.idxVTRec     = 1 : 5;      % SVD termporal patterns for reconstruction

end

%% NLSA MODEL

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
for iC = 1 : In.nC
    In.Trg( iC ).nD = ...
    numel( In.Trg( iC ).xLim( 1 ) : In.Trg( iC ).nDX / 2 : In.Trg( iC ).xLim( 2 ) ); 
    if strcmp( In.Trg( iC ).field, 'tb' )
        In.Trg( iC ).nD = In.Trg( iC ).nD ...
        * numel( In.Trg( iC ).yLim( 1 ) : In.Trg( iC ).nDY / 2 : In.Trg( iC ).yLim( 2 ) );
    end
end 
        

%==============================================================================
% Setup nlsaComponent and nlsaProjectedComponent objects 

nlsaPath = fullfile( pwd, 'data', 'nlsa' );
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

%==============================================================================
% Pairwise distances for in-sample data
modeStr = 'implicit';
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
dFunc = nlsaLocalDistanceFunction( 'localDistance', lDist); 
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
model = nlsaModel( 'path',                            nlsaPath, ...
                   'timeFormat',                      In.tFormat, ...
                   'srcTime',                         tNum, ...
                   'sourceComponent',                 srcComponent, ...
                   'embeddingOrigin',                 In.idxT1, ...
                   'embeddingTemplate',               embComponent, ...
                   'embeddingPartition',              embPartition, ...
                   'pairwiseDistanceTemplate',        pDist, ...
                   'symmetricDistanceTemplate',       sDist, ...
                   'diffusionOperatorTemplate',       diffOp, ...
                   'targetComponent',                 trgComponent, ...
                   'targetEmbeddingTemplate',         trgEmbComponent, ...
                   'projectionTemplate',              prjComponent, ...
                   'reconstructionTemplate',          recComponent, ...
                   'linearMapTemplate',               linMap, ...
                   'svdReconstructionTemplate',       svdRecComponent );

