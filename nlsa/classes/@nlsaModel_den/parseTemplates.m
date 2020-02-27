function constrArgs = parseTemplates( varargin ) 
%PARSETEMPLATES Construct property name-value pairs for the nlsaModel_den class
%   constructor from templates.
%
%   The input arguments of parseTemplates are passed as name-value pairs using
%   the syntax:
%
%   constrArgs = parseTemplates( propName1, propVal1, propName2, propVal2, ... ).
%
%   The following input names can be specified in addition to those in the 
%   parent nlsaModel class
%
%   'denComponent': An [ nCD nR ]-sized array of nlsaComponent objects 
%      specifying the data for density estimation. nCD is the number of 
%      components (physical variables) in the density dataset and nR the
%      number of realizations. If 'denComponent' is not specified it is set to
%      the source data in 'srcComponent'.  
%      
%   'denEmbeddingTemplate': An array of nlsaEmbeddedComponent objects 
%      specifying templates according to which the data in denComponent 
%      are to be time-lagged embedded. denEmbeddingTemplate must be either a 
%      scalar or a vector of size [ nCD 1 ]. In the former case, it is 
%      assumed that every component in the dataset should be embedded using 
%      the same template. If 'denEmbeddingTemplate' is not assigned by the 
%      caller, it is set to the default case of no time-lagged embedding.
%
%   'denEmbeddingOrigin': A scalar or vector of size [ 1 nR ] specifying the
%      starting time index in the data for kernel density estimation to perform
%      time-lagged embedding. The embedding origins must be at least as large
%      as the embedding window of each component. If 'embeddingOrigin' is not
%      specified, the time origin for each realization is set to the minimum
%      possible value consistent with the embedding templates.
%
%   'denEmbeddingPartition': An [ 1 nR ]-sized vector, partition, of 
%      nlsaPartition objects specifying how each realization in the density 
%      data is to be partitioned. The number of samples in partition( iR ) 
%      must be equal to nSDE( iR ), where nSDE( iR ) is the number of samples 
%      in the iR-th realization of the density data after time-lagged 
%      embedding. 
%
%   'denEmbeddingPartitionT': As in 'denEmbeddingPartition', but allows for a
%      different "test" partition, operating along the column dimension of the
%      pairwise distance matrix, to accelerate batch-wise pairwise distance
%      calculation for the density data.
%
%   'denEmbeddingPartitionQ': As in 'denEmbeddingPartitionQ', but for a 
%      "query" partition, operating along the row dimension of the pairwise
%      distance matrix. 
%
%   'denPairwiseDistanceTemplate': An nlsaPairwiseDistance object specifying
%      the method and parameters (e.g., number of nearest neighbors) used 
%      to compute the graph edge weights for the density data. If 
%      'denPairwiseDistanceTemplate' is not specified, it is set to the 
%      (Euclidean) norm, and the number of nearest neighnors nN to 1/100 of 
%      standard L2 number of samples in lagged embedding space.
%
%   'densityComponentName': A string which, if defined, is used to replace the
%      default directory name of the pairwise distances for the density data.
%      This option is useful to avoid long directory names in datasets with
%      several components, but may lead to non-uniqueness of the filename
%      structure and overwriting of results. 
%   
%   'densityRealizationName': Similar to 'srcComponentName', but used to 
%      compress the realization-dependent part of the pairwise distance 
%      directory.
%
%   'kernelDensityTemplate': An nlsaKernelDensity object specifying the 
%      kernel density estimation in the model.
%
%   'densityEmbeddingTemplate': An array of nlsaEmbeddedComponent objects 
%      specifying templates according to which the data in denComponent 
%      are to be time-lagged embedded. embComponent must be either a 
%      scalar or a vector of size [ nC 1 ]. In the former case, it is 
%      assumed that every component in the dataset should be embedded using 
%      the same template. If 'denEmbeddingTemplate' is not assigned by the 
%      caller, it is set to the default case of no time-lagged embedding.
%
%   Contact: dimitris@cims.nyu.edu
%
%   Modified 2020/01/28 


%% CONSTRUCTOR PROPERTY LIST
propName = nlsaModel_den.listConstructorProperties; 
propVal  = cell( size( propName ) );
ifProp   = false( size( propName ) );
idxProp  = zeros( size( propName ) );
nProp    = numel( idxProp );

%% SUPERCLASS CONSTRUCTOR ARGUMENTS
% The use of the nlsaModel_base class is intentional here
parentConstrArgs = nlsaModel_base.parseTemplates( varargin{ : } );
for iProp = 1 : 2 : numel( parentConstrArgs )
    switch parentConstrArgs{ iProp }
        case 'srcComponent'
            iSrcComponent = iProp + 1;
        case 'trgComponent'
            iTrgComponent = iProp + 1;
        case 'embComponent'
            iEmbComponent = iProp + 1;
        case 'trgEmbComponent'
            iTrgEmbComponent = iProp + 1;
        case 'embComponentQ'
            iEmbComponentQ = iProp + 1;
        case 'embComponentT'
            iEmbComponentT = iProp + 1;
    end
end
partition = getPartition( parentConstrArgs{ iEmbComponent }( 1, : ) );
nCT = size( parentConstrArgs{ iTrgEmbComponent }, 1 );

%% MODEL PATH
isSet = false;
for i = 1 : 2 : nargin
    if strcmp( varargin{ i }, 'path' )
        if isSet == 1
            error( 'Path argument has been already specified' )
        end
        if ischar( varargin{ i + 1 } )
            modelPath = varargin{ i + 1 };
            isSet = 1;
        else
            error( 'Invalid path specification' )
        end
    end
end
if ~isSet
    modelPath = pwd;
end

%% DENSITY DATA
% Import density data
% Compatibility of the density data with the source data will be determined by the class constructor 
for iProp = 1 : nProp 
    if strcmp( propName{ iProp }, 'denComponent' )
        iDenComponent = iProp;
        break
    end
end
for i = 1 : 2 : nargin
    if strcmp( varargin{ i }, 'denComponent' )
        if ~isempty( propVal{ iDenComponent } )
            error( 'The density components have been already specified' )
        end
        if ~isa( varargin{ i + 1 }, 'nlsaComponent' )
            error( 'The density data must be specified as an array of nlsaComponent objects' )
        end
        propVal{ iDenComponent } = varargin{ i + 1 };
        ifProp( iDenComponent )  = true;
    end
end
if isempty( propVal{ iDenComponent } )
    propVal{ iDenComponent } = parentConstrArgs{ iSrcComponent };
end     
[ nCD, nR ] = size( propVal{ iDenComponent } );


%% DENSITY EMBEDDED DATA
% Parse embedding templates for the density data   
for iProp = 1 : nProp 
    if strcmp( propName{ iProp }, 'denEmbComponent' )
        iDenEmbComponent = iProp;
        break
    end
end
for i = 1 : 2 : nargin
    if strcmp( varargin{ i }, 'denEmbeddingTemplate' )
        if ~isempty( propVal{ iDenEmbComponent } ) 
            error( 'Time-lagged embedding templates for the density data have been already specified' )
        end
        if  isa( varargin{ i + 1 }, 'nlsaEmbeddedComponent' ) ...
            && isscalar( varargin{ i + 1 } )
            propVal{ iDenEmbComponent } = repmat( varargin{ i + 1 }, [ nCD 1 ] );
        elseif    isa( varargin{ i + 1 }, 'nlsaEmbeddedComponent' ) ...
               && isvector( varargin{ i + 1 } ) ...
               && numel( varargin{ i + 1 } ) == nCD
            propVal{ iDenEmbComponent } = varargin{ i + 1 };
            if isrow( propVal{ iDenEmbComponent } )  
                propVal{ iDenEmbComponent } = propval{ iDenEmbComponent }';
            end
        else
            error( 'Time-lagged embedding templates must be specified as a scalar nlsaEmbeddedComponent object, or vector of nlsaEmbeddedComponent objects of size equal to the number of components' )
        end
        ifProp( iDenEmbComponent ) = true;
    end
end
if isempty( propVal{ iDenEmbComponent } )
    propVal{ iDenEmbComponent } = parentConstrArgs{ iEmbComponent }( :, 1 );;
    ifProp( iDenEmbComponent )  = true;
else
    for iC = 1 : nCD
        propVal{ iDenEmbComponent }( iC ) = setDimension( propVal{ iDenEmbComponent }( iC ), ...
                                                          getDimension( propVal{ iDenComponent }( iC ) ) );
    end
end

% Determine minimum embedding origin from template
minEmbeddingOrigin = getMinOrigin( propVal{ iDenEmbComponent } );

% Replicate template to form embeedded component array
propVal{ iDenEmbComponent } = repmat( propVal{ iDenEmbComponent }, [ 1 nR ] );


% Parse embedding origin templates for the density data
isSet = false;
for i = 1 : 2 : nargin
    if strcmp( varargin{ i }, 'denEmbeddingOrigin' )
        if isSet
            error( 'Time-lagged embedding origins for the density data have been already specified' )
        end
        if ispsi( varargin{ i + 1 } )
            embeddingOrigin = repmat( varargin{ i + 1 }, [ 1 nR ] );
        elseif isvector( varargin{ i + 1 } ) && numel( varargin{ i + 1 } ) == nR 
            embeddingOrigin = varargin{ i + 1 };
        end
        isSet = true;
    end
end
if ~isSet
    for i = 1 : 2 : nargin
        if strcmp( varargin{ i }, 'embeddingOrigin' )
            if isSet
                error( 'Time-lagged embedding origins for the density data have been already specified' )
            end
            if ispsi( varargin{ i + 1 } )
                embeddingOrigin = repmat( varargin{ i + 1 }, [ 1 nR ] );
            elseif isvector( varargin{ i + 1 } ) && numel( varargin{ i + 1 } ) == nR 
                embeddingOrigin = varargin{ i + 1 };
            end
            isSet = true;
        end
    end
end
if ~isSet
    embeddingOrigin = minEmbeddingOrigin * ones( 1, nR );
end
for iR = 1 : nR
    if embeddingOrigin( iR ) < minEmbeddingOrigin
        error( 'Time-lagged embedding origin is below minimum value' )
    end
    for iC = 1 : nCD
        propVal{ iDenEmbComponent }( iC, iR ) = setOrigin( propVal{ iDenEmbComponent }( iC, iR ), embeddingOrigin( iR ) );
    end
end

% Determine maximum number of samples in each embedded component
maxNSDRE = zeros( 1, nR );
for iR = 1 : nR
    maxNSDRE( iR ) = getMaxNSample( propVal{ iDenEmbComponent }( :, iR ), ...
                                    getNSample( propVal{ iDenComponent }( :, iR ) ) );
end


% Parse partition templates
isSet = false;
for i = 1 : 2 : nargin
    if strcmp( varargin{ i }, 'denEmbeddingPartition' )
        if isSet
            error( 'Partition templates have been already specified' )
        end
        if isa( varargin{ i + 1 }, 'nlsaPartition' ) ...
           && isscalar( varargin{ i + 1 } )
            denPartition = repmat( varargin{ i + 1 }, [ 1 nR ] );
        elseif isa( varargin{ i + 1 }, 'nlsaPartition' ) ...
               && isvector( varargin{ i + 1 } ) ...
               && numel( varargin{ i + 1 } ) == nR
            denPartition = varargin{ i + 1 };
        else
            error( 'Data partitions must be specified as a scalar nlsaPartition object, or a vector of nlsaPartition objects of size equal to the number of ensemble realizations' )
        end
        isSet = true;
    end
end 
if ~isSet
    if all( getNSample( partition ) <= maxNSDRE )
        denPartition = partition;
    else
        for iR = nR : -1 : 1
            denPartition( iR ) = nlsaPartition( 'nSample', maxNSDRE( iR ) );
        end
    end
end
for iR = 1 : nR
    if getNSample( denPartition( iR ) ) > maxNSDRE( iR )       
         msgStr = [ 'Number of time-lagged embedded samples ', ...
                    int2str( getNSample( partition( iR ) ) ), ...
                    ' is above maximum value ', ...
                    int2str( maxNSDRE( iR ) ) ];
        error( msgStr ) 

    end
    for iC = 1 : nCD
        propVal{ iDenEmbComponent }( iC, iR ) = setPartition( propVal{ iDenEmbComponent }( iC, iR ), denPartition( iR ) );
    end 
end
nSDRE   = getNSample( denPartition ); % Number of samples in each realization after embedding
nSDE = sum( nSDRE );

% Setup embedded component tags, directories, and filenames
for iR = 1 : nR 
    for iC = 1 : nCD
        propVal{ iDenEmbComponent }( iC, iR ) = ...
            setDefaultTag( propVal{ iDenEmbComponent }( iC, iR ) );
        propVal{ iDenEmbComponent }( iC, iR ) = ...
            setComponentTag( ...
                propVal{ iDenEmbComponent }( iC, iR ), ...
                getComponentTag( propVal{ iDenComponent }( iC, 1 ) ) );
        propVal{ iDenEmbComponent }( iC, iR ) = ...
            setRealizationTag( ...
                propVal{ iDenEmbComponent }( iC, iR ), ...
                getRealizationTag( propVal{ iDenComponent }( 1, iR ) ) );

        tag  = getTag( propVal{ iDenEmbComponent }( iC, iR ) );
        pth  = fullfile( modelPath, 'embedded_data', ...
                         strjoin_e( tag, '_' ) );

        propVal{ iDenEmbComponent }( iC, iR ) = ...
            setPath( propVal{ iDenEmbComponent }( iC, iR ), pth );

        propVal{ iDenEmbComponent }( iC, iR ) = ...
            setDefaultSubpath( propVal{ iDenEmbComponent }( iC, iR ) );

        propVal{ iDenEmbComponent }( iC, iR ) = ...
            setDefaultFile( propVal{ iDenEmbComponent }( iC, iR ) );

    end
end
mkdir( propVal{ iDenEmbComponent } )


% If requested, create "test" embedded components for density data

% Parse "test" partition templates for density data
% Test partition must be a coarsening of the partition in the embComponent
% property. 
isSet = false;
for i = 1 : 2 : nargin
    if strcmp( varargin{ i }, 'denEmbeddingPartitionT' )
        if isSet
            error( 'The test partition for the embedded data has been already specified' )
        end
        if ~(    isa( varargin{ i + 1 }, 'nlsaPartition' ) ...
              && isscalar( varargin{ i + 1 } ) ...
              && isFiner( partition, varargin{ i + 1 } ) ) 
           error( 'Test embedded data partition must be specified as a scalar or nlsaPartition object, which is a coarseing of the embedded data partition.' )
        end
        denPartitionT = varargin{ i + 1 };
        isSet = true;
    end
end 

% If test partition was provided, create an upated denEmbComponentT property; 
% otherwise, it will be set to empty by the class constriuctor
if isSet
    for iProp = 1 : nProp
        if strcmp( propName{ iProp }, 'denEmbComponentT' )
            iDenEmbComponentT = iProp;
            break
        end
    end

    if isa( propVal{ iDenEmbComponent }, 'nlsaEmbeddedComponent_xi' )
        propVal{ iDenEmbComponentT }( nCD, 1 ) = nlsaEmbeddedComponent_xi_e();
    else
        propVal{ iDenEmbComponentT }( nCD, 1 ) = nlsaEmbeddedComponent_e();
    end
    propVal{ iDenEmbComponentT } = mergeCol( propVal{ iDenEmbComponentT }, ...
                                          propVal{ iDenEmbComponent }, ...
                                          'partition', denPartitionT ); 
    ifProp( iDenEmbComponentT ) = true;
    % Check if we should compress realization tags
    isSet2 = false;
    for i = 1 : 2 : nargin
        if strcmp( varargin{ i }, 'densityRealizationName' )
            if ~isSet2
                propVal{ iDenEmbComponentT } = setRealizationTag( ...
                    propVal{ iDenEmbComponentT }, varargin{ i + 1 } );
                isSet2 = true;
                break
            else
                error( 'densityRealizationName has been already set' )
            end
        end
    end  
    for iC = 1 : nCD
        tag  = getTag( propVal{ iDenEmbComponentT }( iC ) );
        pth  = fullfile( modelPath, 'embedded_data', ...
                         strjoin_e( tag, '_' ) );
        propVal{ iDenEmbComponentT }( iC ) = ...        
            setPath( propVal{ iDenEmbComponentT }( iC ), pth );

        propVal{ iDenEmbComponentT }( iC ) = ...
            setDefaultSubpath( propVal{ iDenEmbComponentT }( iC ) );

        propVal{ iDenEmbComponentT }( iC ) = ...
            setDefaultFile( propVal{ iDenEmbComponentT }( iC ) );
    end
    mkdir( propVal{ iDenEmbComponentT } )
else
    denPartitionT = nlsaPartition.empty;
end

% If requested, create "query" embedded components for the density data

% Parse "query" partition templates for density data
% Query partition for density data must be a coarsening of the partition in 
% denEmbComponent. 
isSet = false;
for i = 1 : 2 : nargin
    if strcmp( varargin{ i }, 'denEmbeddingPartitionQ' )
        if isSet
            error( 'The query partition for the embedded density data has been already specified' )
        end
        if ~(    isa( varargin{ i + 1 }, 'nlsaPartition' ) ...
              && isscalar( varargin{ i + 1 } ) ...
              && isFiner( partition, varargin{ i + 1 } ) ) 
           error( 'Query partition for embedded density data must be specified as a scalar or nlsaPartition object, which is a coarseing of the embedded data partition.' )
        end
        denPartitionQ = varargin{ i + 1 };
        isSet = true;
    end
end 

for iProp = 1 : nProp
    if strcmp( propName{ iProp }, 'denEmbComponentQ' )
        iDenEmbComponentQ = iProp;
        break
    end
end

% If query partition was provided create an upated denEmbComponentQ object; 
% otherwise set to original embComponent

if isSet
    % Create "query" embedded components for density data

    if isa( propVal{ iDenEmbComponent }, 'nlsaEmbeddedComponent_xi' )
        propVal{ iDenEmbComponentQ }( nCD, 1 ) = nlsaEmbeddedComponent_xi_e();
    else
        propVal{ iDenEmbComponentQ }( nCD, 1 ) = nlsaEmbeddedComponent_e();
    end
    propVal{ iDenEmbComponentQ } = mergeCol( propVal{ iDenEmbComponentQ }, ...
                                          propVal{ iDenEmbComponent }, ...
                                          'partition', denPartitionQ ); 
    ifProp( iDenEmbComponentQ ) = true;
    % Check if we should compress realization tags
    isSet2 = false;
    for i = 1 : 2 : nargin
        if strcmp( varargin{ i }, 'densityRealizationName' )
            if ~isSet2
                propVal{ iDenEmbComponentQ } = setRealizationTag( ...
                    propVal{ iDenEmbComponentQ }, varargin{ i + 1 } );
                isSet2 = true;
                break
            else
                error( 'densityRealizationName has been already set' )
            end
        end
    end  
    for iC = 1 : nCD 
        tag  = getTag( propVal{ iDenEmbComponentQ }( iC ) );
        pth  = fullfile( modelPath, 'embedded_data', ...
                         strjoin_e( tag, '_' ) );
        propVal{ iDenEmbComponentQ }( iC ) = ...        
            setPath( propVal{ iDenEmbComponentQ }( iC ), pth );

        propVal{ iDenEmbComponentQ }( iC ) = ...
            setDefaultSubpath( propVal{ iDenEmbComponentQ }( iC ) );

        propVal{ iDenEmbComponentQ }( iC ) = ...
            setDefaultFile( propVal{ iDenEmbComponentQ }( iC ) );
    end
    mkdir( propVal{ iDenEmbComponentQ } )
else
    denPartitionQ = denPartition;
end


%% PAIRWISE DISTANCE FOR THE DENSITY DATA
% Parse distance template and set distance partition
for iProp = 1 : nProp 
    if strcmp( propName{ iProp }, 'denPairwiseDistance' )
        iDenPDistance = iProp;
        break
    end
end
for i = 1 : 2 : nargin
    if strcmp( varargin{ i }, 'denPairwiseDistanceTemplate' )
        if ~isempty( propVal{ iDenPDistance } )
            error( 'A pairwise distance template for the density data has been already specified' )
        end
        if ~isa( varargin{ i + 1 }, 'nlsaPairwiseDistance' ) 
            error( 'The pairwise distance template must be specified as an nlsaPairwiseDistance object' )
        end
        nDen = numel( varargin{ i + 1 } );
        if nDen ~= 1 ...
           && ~( nDen == nCD && iscolumn( varargin{ i + 1 } ) )
           error( 'The pairwise distance template must be scalar or a column vector of size equal to the number of density components' )
        end
        propVal{ iDenPDistance } = varargin{ i + 1 };
        ifProp( iDenPDistance )  = true;
    end
end
if isempty( propVal{ iDenPDistance } )
    propVal{ iDenPDistance } = nlsaPairwiseDistance( ...
      'nearestNeighbors', round( getNTotalSample( denPartition ) / 10 ) );
    nDen = 1;
    ifProp( iDenPDistance )  = true;
end
%if nDen < nCD
%    propVal{ iDenPDistance } = repmat( propVal{ iDenPDistance }, [ nCD 1 ] );  
%end
for iD = 1 : nDen
    propVal{ iDenPDistance }( iD ) = setPartition( propVal{ iDenPDistance }( iD ), denPartitionQ );
    propVal{ iDenPDistance }( iD ) = setPartitionTest( propVal{ iDenPDistance }( iD ), denPartitionT );
end

% Loop over the density distances
% Set tags and determine distance-specific directories
for iD = 1 : nDen
    tag = getTag( propVal{ iDenPDistance }( iD ) );
    if ~isempty( tag )
        tag = [ tag '_' ];
    end
    propVal{ iDenPDistance }( iD ) = setTag( propVal{ iDenPDistance }( iD ), ...
            [ tag getDefaultTag( propVal{ iDenPDistance }( iD ) ) ] );

    pth = concatenateTags( propVal{ iDenEmbComponent }( iD, : ) );
    isSet = false;
    for i = 1 : 2 : nargin
        if strcmp( varargin{ i }, 'densityComponentName' ) 
            if ~isSet
                pth{ 1 } = varargin{ i + 1 };
                break
            else
                error( 'densityComponentName has been already set' )
            end
        end
    end
    isSet = false;
    for i = 1 : 2 : nargin
        if strcmp( varargin{ i }, 'densityRealizationName' )
            if ~isSet
                pth{ 2 } = varargin{ i + 1 };
                break
            else
                error( 'densityRealizationName has been already set' )
            end
        end  
    end
    pth = strjoin_e( pth, '_' );

    % Assign pairwise distance paths and filenames

    modelPathDD = fullfile( modelPath, 'processed_data', pth, ...
                            getTag( propVal{ iDenPDistance }( iD ) )  );
    propVal{ iDenPDistance }( iD ) = ...
       setPath( propVal{ iDenPDistance }( iD ), modelPathDD );
    propVal{ iDenPDistance }( iD ) = ...
       setDefaultSubpath( propVal{ iDenPDistance }( iD ) ); 
    propVal{ iDenPDistance }( iD ) = ...
       setDefaultFile( propVal{ iDenPDistance }( iD ) );
end
mkdir( propVal{ iDenPDistance } )

%% KERNEL DENSITY ESTIMATOR
% Parse kernel density template 
for iProp = 1 : nProp 
    if strcmp( propName{ iProp }, 'kernelDensity' )
        iDen = iProp;
        break
    end  
end
for i = 1 : 2 : nargin
    if strcmp( varargin{ i }, 'kernelDensityTemplate' )
        if ~isempty( propVal{ iDen } )
            error( 'A kernel density template has been already specified' )
        end
        if isa( varargin{ i + 1 }, 'nlsaKernelDensity' ) ...
           && all( size( varargin{ i + 1 } ) == size( propVal{ iDenPDistance }   ) )
            propVal{ iDen } = varargin{ i + 1 };
        else
            error( 'The kernel density template must be specified as a scalar nlsaKernelDensity object' )
        end
        ifProp( iDen ) = true;
    end
end
if isempty( propVal{ iDen } )
    for iD = nDen : -1 : 1
        propVal{ iDen }( iD ) = nlsaKernelDensity_fb();
    end
    propVal{ iDen } = propVal{ iDen }';
    ifProp( iDen ) = true;
end
for iD = nDen : -1 : 1
    propVal{ iDen }( iD ) = setPartition( propVal{ iDen }( iD ), denPartitionQ );
%propVal{ iDen } = setPartitionTest( propVal{ iDen }, partition );
    tag = getTag( propVal{ iDen }( iD ) );
    if ~isempty( tag )
        tag = [ tag '_' ];
    end
    propVal{ iDen }( iD ) = setTag( propVal{ iDen }( iD ), ...
            [ tag getDefaultTag( propVal{ iDen }( iD ) ) ] ); 

    % Assign kernel density paths and filenames
    modelPathDD = fullfile( getPath( propVal{ iDenPDistance }( iD ) ) );
    modelPathDL = fullfile( modelPathDD, getTag( propVal{ iDen }( iD ) ) );
    propVal{ iDen }( iD ) = setDefaultSubpath( propVal{ iDen }( iD ) );
    propVal{ iDen }( iD ) = setPath( propVal{ iDen }( iD ), modelPathDL );
    mkdir( propVal{ iDen }( iD ) )
    propVal{ iDen }( iD ) = setDefaultFile( propVal{ iDen }( iD ) );
end

%% KERNEL DENSITY ESTIMATOR -- QUERY PARTITION
if ifProp( iDenEmbComponentQ );
    for iProp = 1 : nProp
        if strcmp( propName{ iProp }, 'kernelDensityQ' )
            iDenQ = iProp;
            break
        end
    end
    for iR = nR : -1 : 1
        for iD = nDen : -1 : 1
            propVal{ iDenQ }( iD, iR ) = nlsaComponent( ...
                 'partition', denPartition( iR ), ...  
                 'componentTag', getTag( propVal{ iDen }( iD ) ), ...
                 'realizationTag', getRealizationTag( propVal{ iDenEmbComponent }( 1, iR ) ) );
            tag  = getTag( propVal{ iDenQ }( iD, iR ) );
            pth  = fullfile( getPath( propVal{ iDen }( iD ) ), ...
                             strjoin_e( tag, '_' ) );
            propVal{ iDenQ }( iD, iR ) = setPath( propVal{ iDenQ }( iD, iR ), ...
                                                  pth );
            propVal{ iDenQ }( iD, iR ) = setDefaultFile( propVal{ iDenQ }( iD, iR ) ); 
            propVal{ iDenQ }( iD, iR ) = setDefaultSubpath( propVal{ iDenQ }( iD, iR ) ); 
            mkdir( propVal{ iDenQ }( iD, iR ) )
        end
    end
    ifProp( iDenQ ) = true;
end

%% DELAY-EMBEDDED DENSITY DATA
% Parse embedding templates   
for iProp = 1 : nProp
    if strcmp( propName{ iProp }, 'embKernelDensity' )
        iEmbDensity = iProp;
        break
    end
end
for i = 1 : 2 : nargin
    if strcmp( varargin{ i }, 'densityEmbeddingTemplate' )
        if ~isempty( propVal{ iEmbDensity } ) 
            error( 'Time-lagged embedding templates have been already specified' )
        end
        if  isa( varargin{ i + 1 }, 'nlsaEmbeddedComponent' ) ...
            && isscalar( varargin{ i + 1 } ) && nC > 1
            propVal{ iEmbDensity } = repmat( varargin{ i + 1 }, [ nC 1 ] );
        elseif    isa( varargin{ i + 1 }, 'nlsaEmbeddedComponent' ) ...
               && isvector( varargin{ i + 1 } ) ...
               && numel( varargin{ i + 1 } ) == nC
            propVal{ iEmbDensity } = varargin{ i + 1 };
            if size( propVal{ iEmbDensity }, 2 ) > 1 
                propVal{ iEmbDensity } = propVal{ iEmbDensity }';
            end
        else
            error( 'Time-lagged embedding templates must be specified as a scalar nlsaEmbeddedComponent object, or vector of nlsaEmbeddedComponent objects of size equal to the number of components' )
        end
        ifProp( iEmbDensity )  = true;
    end
end
if isempty( propVal{ iEmbDensity } )
    for iD = nDen : -1 : 1
        propVal{ iEmbDensity }( iD ) = nlsaEmbeddedComponent_e();
    end
    propVal{ iEmbDensity } = propVal{ iEmbDensity }';
    ifProp( iEmbDensity ) = true;
end

% Determine minimum embedding origin from template
minEmbeddingOrigin = getMinOrigin( propVal{ iEmbDensity } );

% Replicate template to form embedded component array
propVal{ iEmbDensity } = repmat( propVal{ iEmbDensity }, [ 1 nR ] );             


% Parse embedding origin templates
isSet = false;
for i = 1 : 2 : nargin
    if strcmp( varargin{ i }, 'densityEmbeddingOrigin' )
        if isSet
            error( 'Time-lagged embedding origins have been already specified' )
        end
        if ispsi( varargin{ i + 1 } )
            embeddingOrigin = repmat( varargin{ i + 1 }, [ 1 nR ] );
        elseif isvector( varargin{ i + 1 } ) && numel( varargin{ i + 1 } ) == nR 
            embeddingOrigin = varargin{ i + 1 };
        end
        isSet = true;
    end
end
if ~isSet
    embeddingOrigin = minEmbeddingOrigin * ones( 1, nR );
end
for iR = 1 : nR
    for iD = 1 : nDen
        propVal{ iEmbDensity }( iD, iR ) = setOrigin( propVal{ iEmbDensity }( iD, iR ), embeddingOrigin( iR ) );
    end
end

% Determine maximum number of samples in each realization after embedding
maxNSRE = zeros( 1, nR );
for iR = 1 : nR
    maxNSRE( iR ) = getMaxNSample( propVal{ iEmbDensity }( :, iR ), ...
                                   getNSample( parentConstrArgs{ iSrcComponent }( :, iR ) ) );
end

% Assign partitions
for iR = 1 : nR
    if getNSample( partition( iR ) ) > maxNSRE( iR )
        error( 'Number of time-lagged embedded samples is above maximum value' )
    end
    for iD = 1 : nDen
        propVal{ iEmbDensity }( iD, iR ) = setPartition( propVal{ iEmbDensity }( iD, iR ), partition( iR ) );
    end 
end
nSRE   = getNSample( partition ); % Number of samples in each realization after embedding
nSE = sum( nSRE );


% Setup embedded component tags, directories, and filenames
for iR = 1 : nR
     
    for iD = 1 : nDen

        propVal{ iEmbDensity }( iD, iR ) = ...
            setDefaultTag( propVal{ iEmbDensity }( iD, iR ) );
        propVal{ iEmbDensity }( iD, iR ) = ...
            setComponentTag( ...
                propVal{ iEmbDensity }( iD, iR ), ...
                getTag( propVal{ iDen }( iD, 1 ) ) );
        propVal{ iEmbDensity }( iD, iR ) = ...
            setRealizationTag( ...
                propVal{ iEmbDensity }( iD, iR ), ...
                getRealizationTag( ...
                   parentConstrArgs{ iSrcComponent }( 1, iR ) ) );

        tag  = getTag( propVal{ iEmbDensity }( iD, iR ) );
        pth  = fullfile( getPath( propVal{ iDen }( iD ) ), ...
                         'embedded_density', ...
                         strjoin_e( tag, '_' ) );
        
        propVal{ iEmbDensity }( iD, iR ) = ...        
            setPath( propVal{ iEmbDensity }( iD, iR ), pth );

        propVal{ iEmbDensity }( iD, iR ) = ...
            setDefaultSubpath( propVal{ iEmbDensity }( iD, iR ) );


        propVal{ iEmbDensity }( iD, iR ) = ...
            setDefaultFile( propVal{ iEmbDensity }( iD, iR ) );

    end
end
mkdir( propVal{ iEmbDensity } )
          
% If requested, create "test" embedded density

% Parse "test" partition templates.
% Test partition must be a coarsening of the partition in the embComponent 
% property.
isSet = false;
for i = 1 : 2 : nargin
    if strcmp( varargin{ i }, 'embeddingPartitionT' )
        if isSet
            error( 'The test partition for the embedded data has been already specified' )
        end
        if ~(    isa( varargin{ i + 1 }, 'nlsaPartition' ) ...
              && isscalar( varargin{ i + 1 } ) ...
              && isFiner( partition, varargin{ i + 1 } ) ) 
           error( 'Test embedded data partition must be specified as a scalar nlsaPartition object, which is a coarseing of the embedded data partition.' )
        end
        partitionT = varargin{ i + 1 };
        isSet = true;
    end
end 

% If test partition was provided, create an embDensityT property; otherwise,
% it will be set it to empty by the class constructor.
if isSet
    for iProp = 1 : nProp
        if strcmp( propName{ iProp }, 'embKernelDensityT' )
            iEmbDensityT = iProp;
            break
        end
    end

    propVal{ iEmbDensityT }( nDen, 1 ) = nlsaEmbeddedComponent_e();
    propVal{ iEmbDensityT } = mergeCol( propVal{ iEmbDensityT }, ...
                                        propVal{ iEmbDensity }, ...
                                        'partition', partitionT ); 
    ifProp( iEmbDensityT ) = true;
    % Check if we should compress realization tags
    isSet2 = false;
    for i = 1 : 2 : nargin
        if strcmp( varargin{ i }, 'densityRealizationName' )
            if ~isSet2
                propVal{ iEmbDensityT } = setRealizationTag( ...
                    propVal{ iEmbDensityT }, varargin{ i + 1 } );
                isSet2 = true;
                break
            else
                error( 'densityRealizationName has been already set' )
            end
        end
    end  
    for iD = 1 : nDen 
        tag  = getTag( propVal{ iEmbDensityT }( iD ) );
        pth  = fullfile( getPath( propVal{ iDen }( iD ) ), ...
                         'embedded_density', ...
                         strjoin_e( tag, '_' ) );
        propVal{ iEmbDensityT }( iD ) = ...        
            setPath( propVal{ iEmbDensityT }( iD ), pth );
        propVal{ iEmbDensityT }( iD ) = setDefaultSubpath( ...
            propVal{ iEmbDensityT }( iD ) ); 
        propVal{ iEmbDensityT }( iD ) = setDefaultFile( ...
            propVal{ iEmbDensityT }( iD ) );
    end
    mkdir( propVal{ iEmbDensityT } )
else
    partitionT = nlsaPartition.empty;
end

% If requested, create "query" embedded density

% Parse "query" partition templates.
% Query partition must be a coarsening of the partition in the embComponent 
% property.
isSet = false;
for i = 1 : 2 : nargin
    if strcmp( varargin{ i }, 'embeddingPartitionQ' )
        if isSet
            error( 'The query partition for the embedded data has been already specified' )
        end
        if ~(    isa( varargin{ i + 1 }, 'nlsaPartition' ) ...
              && isscalar( varargin{ i + 1 } ) ...
              && isFiner( partition, varargin{ i + 1 } ) ) 
           error( 'Query embedded data partition must be specified as a scalar nlsaPartition object, which is a coarseing of the embedded data partition.' )
        end
        partitionQ = varargin{ i + 1 };
        isSet = true;
    end
end 

% If query partition was provided, create an embDensityQ property; otherwise,
% it will be set it to empty by the class constructor.
if isSet
    for iProp = 1 : nProp
        if strcmp( propName{ iProp }, 'embKernelDensityQ' )
            iEmbDensityQ = iProp;
            break
        end
    end

    propVal{ iEmbDensityQ }( nDen, 1 ) = nlsaEmbeddedComponent_e();
    propVal{ iEmbDensityQ } = mergeCol( propVal{ iEmbDensityQ }, ...
                                        propVal{ iEmbDensity }, ...
                                        'partition', partitionQ ); 
    ifProp( iEmbDensityQ ) = true;
    % Check if we should compress realization tags
    isSet2 = false;
    for i = 1 : 2 : nargin
        if strcmp( varargin{ i }, 'densityRealizationName' )
            if ~isSet2
                propVal{ iEmbDensityQ } = setRealizationTag( ...
                    propVal{ iEmbDensityQ }, varargin{ i + 1 } );
                isSet2 = true;
                break
            else
                error( 'densityRealizationName has been already set' )
            end
        end
    end  
    for iD = 1 : nDen 
        tag  = getTag( propVal{ iEmbDensityQ }( iD ) );
        pth  = fullfile( getPath( propVal{ iDen }( iD ) ), ...
                         'embedded_density', ...
                         strjoin_e( tag, '_' ) );
        propVal{ iEmbDensityQ }( iD ) = ...        
            setPath( propVal{ iEmbDensityQ }( iD ), pth );
        propVal{ iEmbDensityQ }( iD ) = setDefaultSubpath( ...
            propVal{ iEmbDensityQ }( iD ) ); 
        propVal{ iEmbDensityQ }( iD ) = setDefaultFile( ...
            propVal{ iEmbDensityQ }( iD ) );
    end
    mkdir( propVal{ iEmbDensityQ } )
else
    partitionQ = partition;
end




%% PAIRWISE DISTANCE
% Parse distance template and set distance partition
for iProp = 1 : nProp 
    if strcmp( propName{ iProp }, 'pairwiseDistance' )
        iPDistance = iProp;
        break
    end
end
for i = 1 : 2 : nargin
    if strcmp( varargin{ i }, 'pairwiseDistanceTemplate' )
        if ~isempty( propVal{ iPDistance } )
            error( 'A pairwise distance template has been already specified' )
        end
        if isa( varargin{ i + 1 }, 'nlsaPairwiseDistance' ) ...
           && isscalar( varargin{ i + 1 } )
            propVal{ iPDistance } = varargin{ i + 1 };
            ifProp( iPDistance )  = true;
        else
            error( 'The pairwise distance template must be specified as a scalar nlsaPairwiseDistance object' )
        end
    end
end
if isempty( propVal{ iPDistance } )
    propVal{ iPDistance } = nlsaPairwiseDistance( ...
      'distanceFunction', nlsaLocalDistanceFunction_scl(), ...
      'nearestNeighbors', round( getNTotalSample( partition ) / 10 ) );
    ifProp( iPDistance )  = true;
end
nN = getNNeighbors( propVal{ iPDistance } );

propVal{ iPDistance } = setPartition( propVal{ iPDistance }, partitionQ );
propVal{ iPDistance } = setPartitionTest( propVal{ iPDistance }, partitionT );
tag = getTag( propVal{ iPDistance } );
if ~isempty( tag )
    tag = [ tag '_' ];
end
propVal{ iPDistance } = setTag( propVal{ iPDistance }, ...
        [ tag getDefaultTag( propVal{ iPDistance } ) ] ); 

% Determine distance-specific directories
pth = concatenateTags( parentConstrArgs{ iEmbComponent } );
isSet = false;
for i = 1 : 2 : nargin
    if strcmp( varargin{ i }, 'sourceComponentName' ) 
        if ~isSet
            pth{ 1 } = varargin{ i + 1 };
            break
        else
            error( 'sourceComponentName has been already set' )
        end
    end  
end
isSet = false;
for i = 1 : 2 : nargin
    if strcmp( varargin{ i }, 'sourceRealizationName' )
        if ~isSet
            pth{ 2 } = varargin{ i + 1 };
            break
        else
            error( 'sourceRealizationName has been already set' )
        end
    end  
end
pth = strjoin_e( pth, '_' );

pthDen = concatenateTags( propVal{ iDenEmbComponent } );
isSet = false;
for i = 1 : 2 : nargin
    if strcmp( varargin{ i }, 'densityComponentName' ) 
        if ~isSet
            pthDen{ 1 } = varargin{ i + 1 };
            break
        else
            error( 'densityComponentName has been already set' )
        end
    end  
end
isSet = false;
for i = 1 : 2 : nargin
    if strcmp( varargin{ i }, 'densityRealizationName' )
        if ~isSet
            pthDen{ 2 } = varargin{ i + 1 };
            break
        else
            error( 'densityRealizationName has been already set' )
        end
    end  
end
pthDen = strjoin_e( pthDen, '_' );

isSet = false;
for i = 1 : 2 : nargin
    if strcmp( varargin{ i }, 'denDistanceName' )
        if ~isSet
            pthDist = varargin{ i + 1 };
            break
        else
            error( 'denDistanceName has been already set' )
        end
    end  
end
if ~isSet
    pthDist = concatenateTags( propVal{ iDenPDistance } );
end

pthDensity = concatenateTags( propVal{ iEmbDensity } );
%isSet = false;
%for i = 1 : 2 : nargin
%    if strcmp( varargin{ i }, 'embDensityComponentName' ) 
%        if ~isSet
%            pthDensity{ 1 } = varargin{ i + 1 };
%            break
%        else
%            error( 'embDensityComponentName has been already set' )
%        end
%    end  
%end
%isSet = false;
%for i = 1 : 2 : nargin
%    if strcmp( varargin{ i }, 'embDensityRealizationName' )
%        if ~isSet
%            pthDensity{ 2 } = varargin{ i + 1 };
%            break
%        else
%            error( 'embDensityRealizationName has been already set' )
%        end
%    end  
%end

% We only keep component- and embedding-specific tags at this level
pthDensity = strjoin_e( pthDensity( [ 1 3 ] ), '_' );

% Assign pairwise distance paths and filenames
modelPathD = fullfile( modelPath, 'processed_data_den', pth, ...
                       pthDen, pthDist, pthDensity, ...
                       getTag( propVal{ iPDistance } )  );
propVal{ iPDistance } = setPath( propVal{ iPDistance }, modelPathD );
propVal{ iPDistance } = setDefaultSubpath( propVal{ iPDistance } ); 
propVal{ iPDistance } = setDefaultFile( propVal{ iPDistance } );
mkdir( propVal{ iPDistance } )

%% SYMMETRIC DISTANCE
% Parse symmetric distance template 
for iProp = 1 : nProp 
    if strcmp( propName{ iProp }, 'symmetricDistance' )
        iSDistance = iProp;
        break
    end  
end
for i = 1 : 2 : nargin
    if strcmp( varargin{ i }, 'symmetricDistanceTemplate' )
        if ~isempty( propVal{ iSDistance } )
            error( 'A symmetric distance template has been already specified' )
        end
        if isa( varargin{ i + 1 }, 'nlsaSymmetricDistance' ) ...
           && isscalar( varargin{ i + 1 } )
            propVal{ iSDistance } = varargin{ i + 1 };
            ifProp( iSDistance ) = true;
        else
            error( 'The symmetric distance template must be specified as a scalar nlsaSymmetricDistance object' )
        end
    end
end

if isempty( propVal{ iSDistance } )
    propVal{ iSDistance } = nlsaSymmetricDistance_gl( 'nearestNeighbors', nN );
    ifProp( iSDistance ) = true;
end

if getNNeighbors( propVal{ iSDistance } ) > nN
    error( 'The number of nearest neighbors in the symmetric distance matrix cannot exceed the number of neareast neighbors in the pairwise (non-symmetric) distance matrix' )
end
propVal{ iSDistance } = setPartition( propVal{ iSDistance }, partition );
tag = getTag( propVal{ iSDistance } );
if ~isempty( tag )
    tag = [ tag '_' ];
end
propVal{ iSDistance } = setTag( propVal{ iSDistance }, ...
        [ tag getDefaultTag( propVal{ iSDistance } ) ] ); 

% Assign symmetric distance paths and filenames
modelPathS            = fullfile( modelPathD, getTag( propVal{ iSDistance } ) );
propVal{ iSDistance } = setPath( propVal{ iSDistance }, modelPathS );
propVal{ iSDistance } = setDefaultSubpath( propVal{ iSDistance } );
propVal{ iSDistance } = setDefaultFile( propVal{ iSDistance } );
mkdir( propVal{ iSDistance } )



%% DIFFUSION OPERATOR
% Parse diffusion operator template 
for iProp = 1 : nProp 
    if strcmp( propName{ iProp }, 'diffusionOperator' )
        iDiffOp = iProp;
        break
    end  
end
for i = 1 : 2 : nargin
    if strcmp( varargin{ i }, 'diffusionOperatorTemplate' )
        if ~isempty( propVal{ iDiffOp } )
            error( 'A diffusion operator template has been already specified' )
        end
        if isa( varargin{ i + 1 }, 'nlsaDiffusionOperator' ) ...
           && isscalar( varargin{ i + 1 } )
            propVal{ iDiffOp } = varargin{ i + 1 };
        else
            error( 'The diffusion operator template must be specified as a scalar nlsaDiffusionOperator object' )
        end
        ifProp( iDiffOp ) = true;
    end
end
if isempty( propVal{ iDiffOp } )
    propVal{ iDiffOp } = nlsaDiffusionOperator_gl();
    ifProp( iDiffOp ) = true;
end
propVal{ iDiffOp } = setPartition( propVal{ iDiffOp }, partition );
propVal{ iDiffOp } = setPartitionTest( propVal{ iDiffOp }, partition );
if isa( propVal{ iDiffOp }, 'nlsaDiffusionOperator_batch' )
    propVal{ iDiffOp } = setNNeighbors( propVal{ iDiffOp }, ...
                                        getNNeighborsMax( propVal{ iSDistance } ) );
end
nPhi   = getNEigenfunction( propVal{ iDiffOp } );
tag = getTag( propVal{ iDiffOp } );
if ~isempty( tag )
    tag = [ tag '_' ];
end
propVal{ iDiffOp } = setTag( propVal{ iDiffOp }, ...
        [ tag getDefaultTag( propVal{ iDiffOp } ) ] ); 

% Assign diffusion operator  paths and filenames
modelPathL = fullfile( modelPathS, getTag( propVal{ iDiffOp } ) );
propVal{ iDiffOp } = setDefaultSubpath( propVal{ iDiffOp } );
propVal{ iDiffOp } = setPath( propVal{ iDiffOp }, modelPathL );
mkdir( propVal{ iDiffOp } )
propVal{ iDiffOp } = setDefaultFile( propVal{ iDiffOp } );


%% PROJECTED DATA

% Parse projection templates
for iProp = 1 : nProp 
    if strcmp( propName{ iProp }, 'prjComponent' )
        iPrjComponent = iProp;
        break
    end
end
for i = 1 : 2 : nargin
    if strcmp( varargin{ i }, 'projectionTemplate' )
        if ~isempty( propVal{ iPrjComponent } )
            error( 'A projection template has been already specified' )
        end
        if isa( varargin{ i + 1 }, 'nlsaProjectedComponent' ) ...
            && isscalar( varargin{ i + 1 } ) 
            propVal{ iPrjComponent } = repmat( varargin{ i + 1 }, [ nCT 1 ] );
        elseif isa( varargin{ i + 1 }, 'nlsaProjectedComponent' ) ...
           && isvector( varargin{ i + 1 } ) ...
           && numel( varargin{ i + 1 } ) == nCT
            propVal{ iPrjComponent } = varargin{ i + 1 };
            if isrow( propVal{ iPrjComponent } )
                propVal{ iPrjComponent } = propVal{ iPrjComponent }';
            end
        else
            error( 'The projection template must be specified as a scalar nlsaProjectedComponent object, or vector of nlsaProjectedComponent objects of size equal to the number of target components' )
        end
        ifProp( iPrjComponent ) = true;
    end
end
if isempty( propVal{ iPrjComponent } )
    for iC = nCT : -1 : 1
        propVal{ iPrjComponent }( iC ) = nlsaProjectedComponent( 'nBasisFunction', nPhi );
    end
    ifProp( iPrjComponent ) = true;
end

% Setup collective tags, dimension, and partition for projected data
for iC = 1 : nCT
    pth = concatenateTags( parentConstrArgs{ iTrgEmbComponent }( iC, : ) );
    isSet = false;
    for i = 1 : 2 : nargin
        if strcmp( varargin{ i }, 'targetRealizationName' )
            if ~isSet
                pth{ 2 } = varargin{ i + 1 };
                isSet = true;
                break
            else
                error( 'Target realization name has been already specified' )
            end      
        end  
    end
    pth = strjoin_e( pth, '_' );

    propVal{ iPrjComponent }( iC ) = setPartition( ...
        propVal{ iPrjComponent }( iC ), partition );
    propVal{ iPrjComponent }( iC ) = setPath( ...
        propVal{ iPrjComponent }( iC ), fullfile( modelPathL, pth ) );
    propVal{ iPrjComponent }( iC ) = setEmbeddingSpaceDimension( ...
       propVal{ iPrjComponent }( iC ), ...
       getEmbeddingSpaceDimension( ...
           parentConstrArgs{ iTrgEmbComponent }( iC, 1 ) ) );
    propVal{ iPrjComponent }( iC ) = setDefaultSubpath( ...
        propVal{ iPrjComponent }( iC ) );
    propVal{ iPrjComponent }( iC ) = setDefaultFile( propVal{ iPrjComponent }( iC ) );
end
if ~isCompatible( propVal{ iPrjComponent } )
    error( 'Incompatible projection components' )
end
if ~isCompatible( propVal{ iPrjComponent }, propVal{ iDiffOp } )
    error( 'Incompatible projection components and diffusion operator' )
end
mkdir( propVal{ iPrjComponent } )

%% RECONSTRUCTED COMPONENTS

% Parse reconstruction templates
for iProp = 1 : nProp 
    if strcmp( propName{ iProp }, 'recComponent' )
        iRecComponent = iProp;
        break
    end
end
for i = 1 : 2 : nargin
    if strcmp( varargin{ i }, 'reconstructionTemplate' )
        if ~isempty( propVal{ iRecComponent } )
            error( 'A reconstruction template has been already specified' )
        end
        if isa( varargin{ i + 1 }, 'nlsaComponent_rec_phi' ) ...
            && isscalar( varargin{ i + 1 } ) 
            propVal{ iRecComponent } = repmat( varargin{ i + 1 }, [ nCT nR ] );
        elseif isa( varargin{ i + 1 }, 'nlsaComponent_rec_phi' ) ...
           && isvector( varargin{ i + 1 } ) ...
           && numel( varargin{ i + 1 } ) == nR
            propVal{ iRecComponent } = varargin{ i + 1 };
            if isrow( propVal{ iRecComponent } )
                propVal{ iRecComponent } = propVal{ iRecComponent }';
            end
            propVal{ iRecComponent } = repmat( propVal{ iRecComponent }, ...
                                               [ nCT 1 ] );
        else
            error( 'The reconstruction template must be specified as a scalar nlsaComponent_rec_phi object, or vector of nlsaComponent_rec_phi objects of size equal to the number of realizations' )
        end
        ifProp( iRecComponent ) = true;
    end
end
if isempty( propVal{ iRecComponent } )
    for iR = nR : -1 : 1
        for iC = nCT : -1 : 1
            propVal{ iRecComponent }( iC, iR ) = ...
                nlsaComponent_rec_phi( 'basisFunctionIdx', 1 );
        end
    end
    ifProp( iRecComponent ) = true;
end

% Determine maximum number of samples in each realization of the reconstructed
% data
maxNSRR = zeros( 1, nR );
for iR = 1 : nR
    maxNSRR( iR ) = getNSample( ...
                      parentConstrArgs{ iTrgEmbComponent }( 1, iR )  )...
                  + getEmbeddingWindow( ...
                      parentConstrArgs{ iTrgEmbComponent }( 1, iR ) ) ...
                  - 1;
end

% Parse reconstruction partition templates
isSet = false;
for i = 1 : 2 : nargin
    if strcmp( varargin{ i }, 'reconstructionPartition' )
        if isSet
            error( 'Partition templates for the reconstructed data have been already specified' )
        end
        if isa( varargin{ i + 1 }, 'nlsaPartition' ) ...
           && isscalar( varargin{ i + 1 } )
            recPartition = repmat( varargin{ i + 1 }, [ 1 nR ] );
        elseif isa( varargin{ i + 1 }, 'nlsaPartition' ) ...
               && isvector( varargin{ i + 1 } ) ...
               && numel( varargin{ i + 1 } ) == nR
            recPartition = varargin{ i + 1 };
        else
            error( 'Data partitions must be specified as a scalar nlsaPartition object, or a vector of nlsaPartition objects of size equal to the number of ensemble realizations' )
        end
        isSet = true;
    end
end
if ~isSet
    for iR = nR : -1 : 1
        recPartition( iR ) = nlsaPartition( 'nSample', maxNSRR( iR ) );
    end
end
for iR = 1 : nR
    if getNSample( recPartition( iR ) ) > maxNSRR( iR )
        getNSample( recPartition( iR ) )
        maxNSRR( iR )
        error( 'Number of reconstructed samples is above maximum value' )
    end
    for iC = 1 : nCT
        propVal{ iRecComponent }( iC, iR ) = setPartition( propVal{ iRecComponent }( iC, iR ), recPartition( iR ) );
    end
end

% Setup reconstructed component tags, directories, and filenames
for iR = 1 : nR
    for iC = 1 : nCT
        propVal{ iRecComponent }( iC, iR ) = setDimension( ...
           propVal{ iRecComponent }( iC, iR ), ...
           getDimension( parentConstrArgs{ iTrgComponent }( iC, iR ) ) );
        propVal{ iRecComponent }( iC, iR ) = setDefaultTag( ...
            propVal{ iRecComponent }( iC, iR ) );
        propVal{ iRecComponent }( iC, iR ) = setComponentTag( ...
            propVal{ iRecComponent }( iC, iR ), ...
            getComponentTag( parentConstrArgs{ iTrgComponent }( iC, iR ) ) );
        propVal{ iRecComponent }( iC, iR ) = setRealizationTag( ...
            propVal{ iRecComponent }( iC, iR ), ...
            getRealizationTag( parentConstrArgs{ iTrgComponent }( iC, iR ) ) );    
        propVal{ iRecComponent }( iC, iR ) = setDefaultSubpath( ...
            propVal{ iRecComponent }( iC, iR ) );
        propVal{ iRecComponent }( iC, iR ) = setDefaultFile( ...
            propVal{ iRecComponent }( iC, iR ) );
           
        tg = concatenateTags( parentConstrArgs{ iTrgEmbComponent }( iC ) );
        pth = strjoin_e( tg, '_' );
        pth = fullfile( modelPathL, ...
            pth, ...
            getBasisFunctionTag( propVal{ iRecComponent }( iC, iR ) ) );
        propVal{ iRecComponent }( iC, iR ) = ...
            setPath( propVal{ iRecComponent }( iC, iR ), pth );
    end
end
mkdir( propVal{ iRecComponent } )

%% LINEAR MAPS
% Setup collective tags for target data
pth = concatenateTags( parentConstrArgs{ iTrgEmbComponent } );
isSet = false;
for i = 1 : 2 : nargin
    if strcmp( varargin{ i }, 'targetComponentName' )
        if ~isSet
            pth{ 1 } = varargin{ i + 1};
            break
        else
            error( 'Target component name has been already specified' )
        end      
    end  
end
isSet = false;
for i = 1 : 2 : nargin
    if strcmp( varargin{ i }, 'targetRealizationName' )
        if ~isSet
            pth{ 2 } = varargin{ i + 1 };
            break
        else
            error( 'Target realization name has been already specified' )
        end
    end  
end
pth = strjoin_e( pth, '_' );

% Parse linear map templates 
for iProp = 1 : nProp 
    if strcmp( propName{ iProp }, 'linearMap' )
        iLinMap = iProp;
        break
    end
end
for i = 1 : 2 : nargin
    if strcmp( varargin{ i }, 'linearMapTemplate' )
        if ~isempty( propVal{ iLinMap } )
            error( 'A linear map template has been already specified' )
        end
        if isa( varargin{ i + 1 }, 'nlsaLinearMap' ) ...
          && isvector( varargin{ i + 1 } )
            propVal{ iLinMap } = varargin{ i + 1 };
        else
            error( 'The linear map template must be specified as a vector of nlsaLinearMap_basis objects' )
        end
        ifProp( iLinMap ) = true;
    end
end
if isempty( propVal{ iLinMap } )
    propVal{ iLinMap } = nlsaLinearMap_gl( 'basisFunctionIdx', 1 : nPhi );
    ifProp( iLinMap ) = true;
end
nA = numel( propVal{ iLinMap } );


for iA = 1 : nA
    propVal{ iLinMap }( iA ) = setPartition( propVal{ iLinMap }( iA ), ...
      partition );
    propVal{ iLinMap }( iA ) = setDimension( ...
        propVal{ iLinMap }( iA ), ...
        getEmbeddingSpaceDimension( parentConstrArgs{ iTrgEmbComponent }( iC, 1 ) ) );
    if ~isCompatible( propVal{ iLinMap }( iA ), propVal{ iPrjComponent } )
        error( 'Incompatible linear map and projected components' )
    end
    modelPathA = fullfile( modelPathL, pth, getDefaultTag( propVal{ iLinMap }( iA ) ) );
    propVal{ iLinMap }( iA ) = setPath( propVal{ iLinMap }( iA ), modelPathA );
    propVal{ iLinMap }( iA ) = setDefaultSubpath( propVal{ iLinMap }( iA ) );
    propVal{ iLinMap }( iA ) = setDefaultFile( propVal{ iLinMap }( iA ) );
end
if ~isCompatible( propVal{ iLinMap } )
    error( 'Incompatible linear maps' )
end
mkdir( propVal{ iLinMap } )

%% SVD RECONSTRUCTED COMPONENTS
% Parse reconstruction templates
for iProp = 1 : nProp 
    if strcmp( propName{ iProp }, 'svdRecComponent' )
        iSvdRecComponent = iProp;
        break
    end
end
for i = 1 : 2 : nargin
    if strcmp( varargin{ i }, 'svdReconstructionTemplate' )
        if ~isempty( propVal{ iSvdRecComponent } )
            error( 'An SVD reconstruction template has been already specified' )
        end
        if isa( varargin{ i + 1 }, 'nlsaComponent_rec_phi' ) ...
          && isscalar( varargin{ i + 1 } ) 
            propVal{ iSvdRecComponent } = repmat( varargin{ i + 1 }, ...
                                                  [ nCT nR nA ] );
        elseif isa( varargin{ i + 1 }, 'nlsaComponent_rec_phi' ) ...
          && isvector( varargin{ i + 1 } ) ...
          && numel( varargin{ i + 1 } ) == nA
            propVal{ iSvdRecComponent } = repmat( varargin{ i + 1 }, ....
                                                  [ nCT nR 1 ] );
        elseif isa( varargin{ i + 1 }, 'nlsaComponent_rec_phi' ) ...
          && ismatrix( varargin{ i + 1 } ) ...
          && all( size( varargin{ i + 1 } ) == [ nR nA ] )
            propVal{ iSvdRecComponent } = repmat( varargin{ i + 1 }, ...
                                                  [ nCT 1 1 ] );
        elseif isa( varargin{ i + 1 }, 'nlsaComponent_rec_phi' ) ...
          && numel( size( varargin{ i + 1 } ) ) == 3 ...
          && all( size( varargin{ i + 1 } ) == [ nCT nR nA ] )
            propVal{ iSvdRecComponent } = varargin{ i + 1 };      
        else
            error( 'The SVD reconstruction template must be specified as a scalar nlsaComponent_rec_phi object, a vector of nlsaComponent_rec_phi objects of size equal to the number of linear maps, or a matirx of nlsaComponent_rec_phi objects of size equal to the number of realizations times the number of linear maps.' )
        end
        ifProp( iSvdRecComponent ) = true;
    end
end
if isempty( propVal{ iSvdRecComponent } )
    for iA = nA : -1 : 1
        for iR = nR : -1 : 1
            for iC = nCT : -1 : 1
                propVal{ iSvdRecComponent }( iC, iR, iA ) = ...
                    nlsaComponent_rec_phi( 'basisFunctionIdx', 1 );
            end
        end
    end
    ifProp( iSvdRecComponent ) = true;
end
for iA = 1 : nA
    for iR = 1 : nR
        for iC = 1 : nCT
            propVal{ iSvdRecComponent }( iC, iR, iA ) = setPartition( ...
               propVal{ iSvdRecComponent }( iC, iR, iA ), recPartition( iR ) );
            propVal{ iSvdRecComponent }( iC, iR, iA ) = setDimension( ...
               propVal{ iSvdRecComponent }( iC, iR, iA ), ...
               getDimension( parentConstrArgs{ iTrgComponent }( iC, iR ) ) );
            propVal{ iSvdRecComponent }( iC, iR, iA ) = setDefaultTag( ...
               propVal{ iSvdRecComponent }( iC, iR, iA ) );
            propVal{ iSvdRecComponent }( iC, iR, iA ) = setComponentTag( ...
                propVal{ iSvdRecComponent }( iC, iR, iA ), ...
                getComponentTag( parentConstrArgs{ iTrgComponent }( iC, iR ) ) );
            propVal{ iSvdRecComponent }( iC, iR, iA ) = setRealizationTag( ...
                propVal{ iSvdRecComponent }( iC, iR, iA ), ...
                getRealizationTag( parentConstrArgs{ iTrgComponent }( iC, iR ) ) );    
            propVal{ iSvdRecComponent }( iC, iR, iA ) = setDefaultSubpath( ...
                propVal{ iSvdRecComponent }( iC, iR, iA ) );
            propVal{ iSvdRecComponent }( iC, iR, iA ) = setDefaultFile( ...
                propVal{ iSvdRecComponent }( iC, iR, iA ) );

            tg = concatenateTags( parentConstrArgs{ iTrgEmbComponent }( iC ) );
            pth = strjoin_e( tg, '_' );
            pth = fullfile( ... 
                getPath( propVal{ iLinMap }( iA ) ), ...
                pth, ...
                getBasisFunctionTag( propVal{ iSvdRecComponent }( iC, iR, iA ) ) );
            propVal{ iSvdRecComponent }( iC, iR, iA ) = ...
                setPath( propVal{ iSvdRecComponent }( iC, iR, iA ), pth );
        end
    end
end
mkdir( propVal{ iSvdRecComponent } )


%% COLLECT ARGUMENTS
constrArgs                = cell( 1, 2 * nnz( ifProp ) );
constrArgs( 1 : 2 : end ) = propName( ifProp );
constrArgs( 2 : 2 : end ) = propVal( ifProp );
constrArgs = [ constrArgs parentConstrArgs ];
