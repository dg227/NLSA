function constrArgs = parseTemplates( varargin ) 
%PARSETEMPLATES Construct property name-value pairs for the nlsaModel_ent class
%   constructor from templates.
%
%   The arguments of parseTemplates are passed as property name-property value
%   pairs using the syntax:
%
%   propNameVals  = parseTemplates( propName1, propVal1, propName2, propVal2, ... ).
%
%   The following properties can be specified in addition to those available in the
%   parseTemplates method of the nlsaModel_ose superclass:
%
%   'sclPartition': An [ 1 nRO ]-sized vector of nlsaPartition objects
%      specifying how each realization in the OSE dataset is to be partitioned.
%      The number of samples in partition( iR ) must be equal to nSE( iR ),
%      where nSE( iR ) is the number of samples in the iR-th realization
%      after time lagged embedding. 
%
%   'sclOutPairwiseDistanceTemplate': An nlsaPairwiseDistance_scl object specifying
%      the method and parameters (e.g., number of nearest neighbors) used 
%      to compute the graph edge weights for the dataset. If 
%      'sclOutPairwiseDistanceTemplate' is not specified, it is set to the 
%      (Euclidean) norm, and the number of nearest neighnors nN to 1/100 of 
%      standard L2 number of samples in lagged embedding space.
%
%   'sclOutSymmetricDistanceTemplate': 
%
%   'sclOutDiffusionOperatorTemplate': An nlsaDiffusionOperator object specifying 
%       the diffusion operator for the entropy-weighted distances 

%   See also MAKEEMBEDDING, COMPUTEPAIRWISEDISTANCES, SYMMETRIZEDISTANCES,
%      COMPUTETEMPORALOPERATOR, MAKETARGETEMBEDDING, COMPUTESVD, 
%      COMPUTETEMPORALPATTERNS, COMPUTESPECTRALENTROPY, RECONSTRUCTMODES
%
%   References
%   [1] D. Giannakis and A. J. Majda (2012), "Nonlinear Laplacian spectral 
%      analysis for time series with intermittency and low-frequency 
%      variability", Proc. Natl. Acad. Sci., 109(7), 2222, 
%      doi:10.1073/pnas.1118984109 
%   [2] D. Giannakis and A. J. Majda (2012), "Nonlinear Laplacian spectral 
%      analysis: Capturing intermittent and low-frequency spatiotemporal 
%      patterns in high-dimensional data", Stat. Anal. and Data Min., 
%      doi:10.1002/sam.11171
%
%   Contact: dimitris@cims.nyu.edu
%
%   Modified 2014/10/13    


%% CONSTRUCTOR PROPERTY LIST
propName = nlsaModel_scl.listConstructorProperties;         
propVal  = cell( size( propName ) );
ifProp   = false( size( propName ) );
idxProp  = zeros( size( propName ) );
nProp    = numel( idxProp );

%% SUPERCLASS CONSTRUCTOR ARGUMENTS
parentConstrArgs = nlsaModel_err.parseTemplates( varargin{ : } );
for iProp = 1 : 2 : numel( parentConstrArgs )
    switch parentConstrArgs{ iProp }
        case 'embComponent'
            iEmbComponent = iProp + 1;
        case 'trgEmbComponent'
            iTrgEmbComponent = iProp + 1;
        case 'prjComponent'
            iPrjComponent = iProp + 1;
        case 'oseEmbComponent'
            iOseEmbComponent = iProp + 1;
        case 'outEmbComponent'
            iOutEmbComponent = iProp + 1;
        case 'outTrgEmbComponent'
            iOutTrgEmbComponent = iProp + 1;
        case 'osePairwiseDistance' 
            iOsePDist = iProp + 1;
        case 'diffusionOperator'
            iDiffOp = iProp + 1;
        case 'oseDiffusionOperator'  
            iOseDiffOp = iProp + 1;
        case 'outPairwiseDistance'
            iOutPDist = iProp + 1;
        case 'oseRefComponent'
            iOseRefComponent = iProp + 1;
        case 'isrEmbComponent'
            iIsrEmbComponent = iProp + 1;
        case 'isrRefComponent'
            iIsrRefComponent = iProp + 1;
    end
end 
partition = getPartition( parentConstrArgs{ iEmbComponent } );
[ nCT, nR ] = size( parentConstrArgs{ iTrgEmbComponent } );
[ ~, nRO ]  = size( parentConstrArgs{ iOseEmbComponent } );


%% SCALED PAIRWISE DISTANCE
% Parse scaled distance template and set distance partition
for iProp = 1 : nProp 
    if strcmp( propName{ iProp }, 'sclOutPairwiseDistance' )
        iSclOutPDist = iProp;
        break
    end
end
for i = 1 : 2 : nargin
    if strcmp( varargin{ i }, 'sclOutPairwiseDistanceTemplate' )
        if ~isempty( propVal{ iSclOutPDist } )
            error( 'A scaled pairwise distance template has been already specified' )
        end
        if isa( varargin{ i + 1 }, 'nlsaPairwiseDistance_scl' ) ...
           && isscalar( varargin{ i + 1 } )
            propVal{ iSclOutPDist } = varargin{ i + 1 };
            ifProp( iSclOutPDist )  = true;
        else
            error( 'The scaled pairwise distance template must be specified as a scalar nlsaPairwiseDistance_scl object' )
        end
    end
end
if isempty( propVal{ iSclOutPDist } )
    propVal{ iSclOutPDist } = parentConstrArgs{ iOutPDist };
    ifProp( iSclOutPDist )  = true;
end

propVal{ iSclOutPDist } = setPartition( propVal{ iSclOutPDist }, ...
                                     getPartition( parentConstrArgs{ iOutPDist } ) ); 
propVal{ iSclOutPDist } = setPartitionTest( propVal{ iSclOutPDist }, ...
                                         getPartition( parentConstrArgs{ iOutPDist } ) ); 
tag = getTag( propVal{ iSclOutPDist } );
if ~isempty( tag )
    tag = [ tag '_' ];
end
propVal{ iSclOutPDist } = setTag( propVal{ iSclOutPDist }, ...
        [ tag getDefaultTag( propVal{ iSclOutPDist } ) ] ); 


% Determine distance-specific directories
% Ose components
pthO = concatenateTags( parentConstrArgs{ iOseEmbComponent } );
isSet = false;
for i = 1 : 2 : nargin
    if strcmp( varargin{ i }, 'oseComponentName' ) 
        if ~isSet
            pthO{ 1 } = varargin{ i + 1 };
            break
        else
            error( 'oseComponentName has been already set' )
        end
    end  
end
isSet = false;
for i = 1 : 2 : nargin
    if strcmp( varargin{ i }, 'oseRealizationName' )
        if ~isSet
            pthO{ 2 } = varargin{ i + 1 };
            break
        else
            error( 'oseRealizationName has been already set' )
        end
    end  
end
pthO = strjoin_e( pthO, '_' );
% Reference components
pthOR = concatenateTags( parentConstrArgs{ iOseRefComponent } );
isSet = false;
for i = 1 : 2 : nargin
    if strcmp( varargin{ i }, 'oseReferenceComponentName' ) 
        if ~isSet
            pthOR{ 1 } = varargin{ i + 1 };
            break
        else
            error( 'oseReferenceComponentName has been already set' )
        end
    end  
end
isSet = false;
for i = 1 : 2 : nargin
    if strcmp( varargin{ i }, 'oseReferenceRealizationName' )
        if ~isSet
            pthOR{ 2 } = varargin{ i + 1 };
        end
    end  
end
pthOR = strjoin_e( pthOR, '_' );
if isa( parentConstrArgs{ iOseRefComponent }, 'nlsaEmbeddedComponent_rec' );
    pthOR = concatenateTags( parentConstrArgs{ iOseEmbComponent } );
    isSet = false;
    for i = 1 : 2 : nargin
        if strcmp( varargin{ i }, 'outSourceComponentName' ) 
            if ~isSet
                pthOR{ 1 } = varargin{ i + 1 };
                break
            else
                error( 'outSourceComponentName has been already set' )
            end
        end  
    end
    isSet = false;
    for i = 1 : 2 : nargin
        if strcmp( varargin{ i }, 'outSourceRealizationName' )
            if ~isSet
                pthOR{ 2 } = varargin{ i + 1 };
                break
            else
                error( 'outSourceRealizationName has been already set' )
            end
        end  
    end
    pthOR = strjoin_e( pthOR, '_' );
    pthOR = strjoin_e( pthOR, ... 
                   getTag( parentConstrArgs{ iOutPDistance } ) , ...
                   getTag( parentConstrArgs{ iOytSDistance } ), ...
                   getTag( parentConstrArgs{ iOutDiffOp } ), '_' ); 
end
modelPathDE = fullfile( getPath( parentConstrArgs{ iOseDiffOp } ), ...
                        pthO, ...
                        pthOR, ...
                        getTag( propVal{ iSclOutPDist } ) ); 
propVal{ iSclOutPDist } = setPath( propVal{ iSclOutPDist }, modelPathDE ); 
propVal{ iSclOutPDist } = setDefaultSubpath( propVal{ iSclOutPDist } );
propVal{ iSclOutPDist } = setDefaultFile( propVal{ iSclOutPDist } );
mkdir( propVal{ iSclOutPDist } )
nN   = getNNeighbors( propVal{ iSclOutPDist } );

%% SCALED SYMMETRIC DISTANCE
% Parse symmetric distance template
for iProp = 1 : nProp
    if strcmp( propName{ iProp }, 'sclOutSymmetricDistance' )
        iSclOutSDist = iProp;
        break
    end
end
for i = 1 : 2 : nargin
    if strcmp( varargin{ i }, 'sclOutSymmetricDistanceTemplate' )
        if ~isempty( propVal{ iSclOutSDist } )
            error( 'A scaled symmetric distance template has been already specified' )
        end
        if isa( varargin{ i + 1 }, 'nlsaSymmetricDistance' ) ...
           && isscalar( varargin{ i + 1 } )
            propVal{ iSclOutSDist } = varargin{ i + 1 };
            ifProp( iSclOutSDist ) = true;
        else
            error( 'The symmetric distance template must be specified as a scalar nlsaSymmetricDistance object' )
        end
    end
end

if isempty( propVal{ iSclOutSDist } )
    propVal{ iSclOutSDist } = nlsaSymmetricDistance_gl( 'nearestNeighbors', nN );
    ifProp( iSclOutSDist ) = true;
end
if getNNeighbors( propVal{ iSclOutSDist } ) > nN
    error( 'The number of nearest neighbors in the symmetric distance matrix cannot exceed the number of neareast neighbors in the pairwise (non-symmetric) distance matrix' )
end
propVal{ iSclOutSDist } = setPartition( propVal{ iSclOutSDist }, getPartition( propVal{ iSclOutPDist } ) );
tag = getTag( propVal{ iSclOutSDist } );
if ~isempty( tag )
    tag = [ tag '_' ];
end
propVal{ iSclOutSDist } = setTag( propVal{ iSclOutSDist }, ...
        [ tag getDefaultTag( propVal{ iSclOutSDist } ) ] );

% Assign symmetric distance paths and filenames
modelPathSE          = fullfile( modelPathDE, getTag( propVal{ iSclOutSDist } ) );
propVal{ iSclOutSDist } = setPath( propVal{ iSclOutSDist }, modelPathSE  );
propVal{ iSclOutSDist } = setDefaultSubpath( propVal{ iSclOutSDist } );
propVal{ iSclOutSDist } = setDefaultFile( propVal{ iSclOutSDist } );
mkdir( propVal{ iSclOutSDist } )


%% SCALED DIFFUSION OPERATOR 
% Parse scaled diffusion operator template 
for iProp = 1 : nProp 
    if strcmp( propName{ iProp }, 'sclOutDiffusionOperator' )
        iSclOutDiffOp = iProp;
        break
    end  
end
isSet = false;
for i = 1 : 2 : nargin
    if strcmp( varargin{ i }, 'sclOutDiffusionOperatorTemplate' )
        if ~isSet
            if isa( varargin{ i + 1 }, 'nlsaDiffusionOperator' ) ...
               && isscalar( varargin{ i + 1 } )
                propVal{ iSclOutDiffOp } = varargin{ i + 1 };
                isSet = true;
            else
                error( 'The scaled diffusion operator template must be specified as a scalar nlsaDiffusionOperator object' )
            end
        else       
            error( 'A scaled diffusion operator template has been already specified' )
        end
    end
end
if ~isSet
    propVal{ iSclOutDiffOp } = parentConstrArgs{ iOutDiffOp };
end
ifProp( iSclOutDiffOp )  = true;
propVal{ iSclOutDiffOp } = setPartition( propVal{ iSclOutDiffOp }, ...
                        getPartition( propVal{ iSclOutSDist } ) );
propVal{ iSclOutDiffOp } = setPartitionTest( propVal{ iSclOutDiffOp }, ...
                        getPartition( propVal{ iSclOutSDist } ) );
propVal{ iSclOutDiffOp } = setNNeighbors( propVal{ iSclOutDiffOp }, ...
                                       getNNeighborsMax( propVal{ iSclOutSDist } ) );
tag = getTag( propVal{ iSclOutDiffOp } );
if ~isempty( tag )
    tag = [ tag '_' ];
end
propVal{ iSclOutDiffOp } = setTag( propVal{ iSclOutDiffOp }, ...
        [ tag getDefaultTag( propVal{ iSclOutDiffOp } ) ] ); 

% Assign diffusion operator paths and filenames
modelPathL           = fullfile( getPath( propVal{ iSclOutSDist } ), ...
                                  getTag( propVal{ iSclOutDiffOp } ) );
propVal{ iSclOutDiffOp } = setDefaultSubpath( propVal{ iSclOutDiffOp } );
propVal{ iSclOutDiffOp } = setPath( propVal{ iSclOutDiffOp }, modelPathL );
mkdir( propVal{ iSclOutDiffOp } )
propVal{ iSclOutDiffOp } = setDefaultFile( propVal{ iSclOutDiffOp } );


%% SCALED OS PROJECTED DATA
% Parse projection templates
for iProp = 1 : nProp 
    if strcmp( propName{ iProp }, 'sclOutPrjComponent' )
        iSclOutPrjComponent = iProp;
        break
    end
end
for i = 1 : 2 : nargin
    if strcmp( varargin{ i }, 'sclProjectionTemplate' )
        if ~isempty( propVal{ iSclOutPrjComponent } )
            error( 'A projection template for the model data has been already specified' )
        end
        if isa( varargin{ i + 1 }, 'nlsaProjectedComponent' ) ...
            && isscalar( varargin{ i + 1 } ) 
            propVal{ iSclOutPrjComponent } = repmat( varargin{ i + 1 }, [ nCT 1 ] );
        elseif isa( varargin{ i + 1 }, 'nlsaProjectedComponent' ) ...
           && isvector( varargin{ i + 1 } ) ...
           && numel( varargin{ i + 1 } ) == nCT
            propVal{ iSclOutPrjComponent } = varargin{ i + 1 };
            if isrow( propVal{ iSclOutPrjComponent } )
                propVal{ iSclOutPrjComponent } = propVal{ iSclOutPrjComponent }';
            end
        else
            error( 'The projection template must be specified as a scalar nlsaProjectedComponent object, or vector of nlsaProjectedComponent objects of size equal to the number of target components' )
        end
        ifProp( iSclOutPrjComponent ) = true;
    end
end
if isempty( propVal{ iSclOutPrjComponent } )
    for iC = nCT : -1 : 1
        propVal{ iSclOutPrjComponent }( iC ) = nlsaProjectedComponent( 'nBasisFunction', ...
          getNEigenfunction( propVal{ iSclOutDiffOp } ) );
    end
    ifProp( iSclOutPrjComponent ) = true;
end
for iC = 1 : nCT
    % Setup collective tags for projected data
    isSet = false;
    for i = 1 : 2 : nargin
        if strcmp( varargin{ i }, 'projectionComponentName' )
            if ~isSet
                pth = varargin{ i + 1 };
                break
            else
                error( 'Projection component name has been already specified' )
            end
        end
    end
    if ~isSet
        pth = concatenateTags( parentConstrArgs{ iTrgEmbComponent }( iC, : ) );
        pth = strjoin_e( pth, '_' );
    end
    
    propVal{ iSclOutPrjComponent }( iC ) = setPartition( propVal{ iSclOutPrjComponent }( iC ), partition );
    propVal{ iSclOutPrjComponent }( iC ) = setPath( propVal{ iSclOutPrjComponent }( iC ), ...
    fullfile( modelPathL, pth ) ); 
    propVal{ iSclOutPrjComponent }( iC ) = setEmbeddingSpaceDimension( propVal{ iSclOutPrjComponent }( iC ), ...
      getEmbeddingSpaceDimension( parentConstrArgs{ iOutTrgEmbComponent }( iC ) ) ); 
    propVal{ iSclOutPrjComponent }( iC ) = setDefaultSubpath( propVal{ iSclOutPrjComponent }( iC ) );
    propVal{ iSclOutPrjComponent }( iC ) = setDefaultFile( propVal{ iSclOutPrjComponent }( iC ) );
end
mkdir( propVal{ iSclOutPrjComponent } )


%% SCALED PAIRWISE DISTANCES FOR ISR
% Parse distance template and set partition
for iProp = 1 : nProp 
    if strcmp( propName{ iProp }, 'sclIsrPairwiseDistance' )
        iSclIsrPDist = iProp;
        break
    end
end
for i = 1 : 2 : nargin
    if strcmp( varargin{ i }, 'sclIsrPairwiseDistanceTemplate' )
        if ~isempty( propVal{ iSclIsrPDist } )
            error( 'A pairwise distance template has been already specified' )
        end
        if isa( varargin{ i + 1 }, 'nlsaPairwiseDistance' ) ...
           && isscalar( varargin{ i + 1 } )
            propVal{ iSclIsrPDist } = varargin{ i + 1 };
            ifProp( iSclIsrPDist )  = true;
            ifRetainTag          = true;
        else
            error( 'The OSE pairwise distance template must be specified as a scalar nlsaPairwiseDistance object' )
        end
    end
end
if isempty( propVal{ iSclIsrPDist } )
    propVal{ iSclIsrPDist } = parentConstrArgs{ iOsePDist };
    ifProp( iSclIsrPDist )  = true;
    ifRetainTag          = false;
end

% Partition for query (in-sample) data
propVal{ iSclIsrPDist } = setPartition( propVal{ iSclIsrPDist }, ...
                        getPartitionTest( parentConstrArgs{ iOsePDist } ) );

% partition for test (out-of-sample) data 
propVal{ iSclIsrPDist } = setPartitionTest( propVal{ iSclIsrPDist }, ...
                         getPartition( parentConstrArgs{ iOsePDist } ) ); 

if ifRetainTag
    tag = getTag( propVal{ iSclIsrPDist } );
    if ~isempty( tag )
        tag = [ tag '_' ];
    end
else
    tag = [];
end
propVal{ iSclIsrPDist } = setTag( propVal{ iSclIsrPDist }, ...
        [ tag getDefaultTag( propVal{ iSclIsrPDist } ) ] ); 

% Determine distance-specific directories
% Isr components
pthI = concatenateTags( parentConstrArgs{ iIsrEmbComponent } );
isSet = false;
for i = 1 : 2 : nargin
    if strcmp( varargin{ i }, 'isrComponentName' ) 
        if ~isSet
            pthI{ 1 } = varargin{ i + 1 };
            break
        else
            error( 'isrComponentName has been already set' )
        end
    end  
end
isSet = false;
for i = 1 : 2 : nargin
    if strcmp( varargin{ i }, 'isrRealizationName' )
        if ~isSet
            pthI{ 2 } = varargin{ i + 1 };
            break
        else
            error( 'isrRealizationName has been already set' )
        end
    end  
end
pthI = strjoin_e( pthI, '_' );
% Reference components
pthIR = concatenateTags( parentConstrArgs{ iIsrRefComponent } );
isSet = false;
for i = 1 : 2 : nargin
    if strcmp( varargin{ i }, 'oseReferenceComponentName' ) 
        if ~isSet
            pthIR{ 1 } = varargin{ i + 1 };
            break
        else
            error( 'oseReferenceComponentName has been already set' )
        end
    end  
end
isSet = false;
for i = 1 : 2 : nargin
    if strcmp( varargin{ i }, 'oseReferenceRealizationName' )
        if ~isSet
            pthIR{ 2 } = varargin{ i + 1 };
        end
    end  
end
pthIR = strjoin_e( pthIR, '_' );
if isa( parentConstrArgs{ iOseRefComponent }, 'nlsaEmbeddedComponent_rec' );
    pthIR = concatenateTags( parentConstrArgs{ iOseEmbComponent } );
    isSet = false;
    for i = 1 : 2 : nargin
        if strcmp( varargin{ i }, 'outSourceComponentName' ) 
            if ~isSet
                pthIR{ 1 } = varargin{ i + 1 };
                break
            else
                error( 'outSourceComponentName has been already set' )
            end
        end  
    end
    isSet = false;
    for i = 1 : 2 : nargin
        if strcmp( varargin{ i }, 'outSourceRealizationName' )
            if ~isSet
                pthIR{ 2 } = varargin{ i + 1 };
                break
            else
                error( 'outSourceRealizationName has been already set' )
            end
        end  
    end
    pthIR = strjoin_e( pthIR, '_' );
    pthIR = strjoin_e( pthIR, ... 
                   getTag( parentConstrArgs{ iPDistance } ) , ...
                   getTag( parentConstrArgs{ iSDistance } ), ...
                   getTag( parentConstrArgs{ iDiffOp } ), '_' ); 
end

% Assign pairwise distance paths and filenames
modelPathDE = fullfile( getPath( parentConstrArgs{ iOseDiffOp } ), ...
                        pthO, ...
                        pthOR, ...
                        pthI, ...
                        pthIR, ...
                        getTag( propVal{ iSclIsrPDist } ) ); 
propVal{ iSclIsrPDist } = setPath( propVal{ iSclIsrPDist }, modelPathDE ); 
propVal{ iSclIsrPDist } = setDefaultSubpath( propVal{ iSclIsrPDist } );
propVal{ iSclIsrPDist } = setDefaultFile( propVal{ iSclIsrPDist } );
mkdir( propVal{ iSclIsrPDist } )


%% SCALED ISR DIFFUSION OPERATOR 
for iProp = 1 : nProp 
    if strcmp( propName{ iProp }, 'sclIsrDiffusionOperator' )
        iSclIsrDiffOp = iProp;
        break
    end  
end
isSet = false;
for i = 1 : 2 : nargin
    if strcmp( varargin{ i }, 'sclIsrDiffusionOperatorTemplate' )
        if ~isSet
            if isa( varargin{ i + 1 }, 'nlsaDiffusionOperator_ose' ) ...
               && isscalar( varargin{ i + 1 } )
                propVal{ iSclIsrDiffOp } = varargin{ i + 1 };
                isSet = true;
            else
                error( 'The OSE diffusion operator template must be specified as a scalar nlsaDiffusionOperator_ose object' )
            end
        else       
            error( 'An OSE diffusion operator template has been already specified' )
        end
    end
end
if ~isSet
    propVal{ iSclIsrDiffOp } = nlsaDiffusionOperator_ose( parentConstrArgs{ iDiffOp } );
end
ifProp( iSclIsrDiffOp )  = true;
propVal{ iSclIsrDiffOp } = setPartition( propVal{ iSclIsrDiffOp }, ...
                                    getPartition( propVal{ iSclIsrPDist } ) );
propVal{ iSclIsrDiffOp } = setPartitionTest( propVal{ iSclIsrDiffOp }, ...
                                    getPartitionTest( propVal{ iSclIsrPDist } ) );
tag = getTag( propVal{ iSclIsrDiffOp } );
if ~isempty( tag )
    tag = [ tag '_' ];
end
propVal{ iSclIsrDiffOp } = setTag( propVal{ iSclIsrDiffOp }, ...
        [ tag getDefaultTag( propVal{ iSclIsrDiffOp } ) ] ); 

% Assign diffusion operator paths and filenames
modelPathLO           = fullfile( getPath( propVal{ iSclIsrPDist } ), ...
                                  getTag( propVal{ iSclIsrDiffOp } ) );
propVal{ iSclIsrDiffOp } = setDefaultSubpath( propVal{ iSclIsrDiffOp } );
propVal{ iSclIsrDiffOp } = setPath( propVal{ iSclIsrDiffOp }, modelPathLO );
mkdir( propVal{ iSclIsrDiffOp } )
propVal{ iSclIsrDiffOp } = setDefaultFile( propVal{ iSclIsrDiffOp } );

% COLLECT CONSTRUCTOR ARGUMENTS
constrArgs                = cell( 1, 2 * nnz( ifProp ) );
constrArgs( 1 : 2 : end ) = propName( ifProp );
constrArgs( 2 : 2 : end ) = propVal( ifProp );
constrArgs = [ constrArgs parentConstrArgs ];
