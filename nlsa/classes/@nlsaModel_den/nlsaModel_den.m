classdef nlsaModel_den < nlsaModel
%NLSAMODEL_DEN   Class definition and constructor of NLSA model with
%  kernel density estimation
%
%  nlsaModel_den builds on the nlsaModel parent class to implement variable
%  bandwidth kernels [1] where the bandwidth is a function of the density of 
%  the sampling measure of the data relative to a kernel-induced measure (e.g.,
%  the Riemannian measure for kernels normalized via the diffusion maps 
%  normalization).  
%  
%  The nlsaModel_den class implements methods for the following operations:
%
%  (i) Delay-embedding of data used for kernel density estimation and the
%      corresponding phase space velocities.
%
%  (ii) Calculation of the pairwise distances between the data in (i).
%
%  (iii) Kernel density estimation using the distances in (ii).
%
%  The kernel density estimates from (iii) are then employed in the 
%  computation of the pairwise distances between the in-sample data
%
%  The class constructor arguments are passed as property name-property value
%  pairs using the syntax
%
%  model = nlsaModel( propName1, propVal1, propName2, propVal2, ... ).
%
%  In addition to the of the nlsaModel_base parent class, the following 
%  properties can be specified:
%
%  'denComponent': An [ nC nR ]-sized array of nlsaComponent objects 
%     specifying the source data. nC is the number of components (physical
%     variables) in the dataset and nR the number of realizations (ensemble 
%     members) as specified in the srcComponent property of the parent 
%     nlsaModel_base class. If 'denComponent' is not specified it is set equal
%     to srcComponent.
%
%  'denEmbComponent': An [ nC nR ]-sized array of nlsaEmbeddedComponent 
%     objects storing Takens delay-embedded data associacted with 
%     'denComponent'. See the documentation of the nlsaModel_base class for 
%     additional information. If 'denEmbComponent' is not specified
%     it is set equal to property embComponent of the nlsaModel_base parent 
%     class. The latter is the most usual case in practice. 
%
%   'denEmbComponentT': An [ nC 1 ]-sized array (column vector) of 
%      nlsaEmbeddedComponent_e  objects, storing a coarsened partition of the 
%      embedded data for density estimation. denEmbComponentT can be used to 
%      accelerate the batchwise pairwise distance calculation in the column 
%      (test) dimension. If not set by the user, denEmbComponentT is set by 
%      default to denEmbComponent.
%
%   'denEmbComponentQ': As in 'denEmbComponentT', but an [ nC 1 ]-sized array
%      of nlsaEmbeddedComponent_e objects to accelerate the pairwise distance
%      calculation in the row (query) dimension.
%
%   'denPairwiseDistance': An nlsaPairwiseDistance object specifying
%      the method and parameters (e.g., number of nearest neighbors) used 
%      to compute the pairwise distances for the delay-embedded in 
%      denEmbComponent.  nlsaPairwiseDistance objects have a property
%      partition which must be set to a vector of nlsaPartition objects of size
%      [ 1 nR ], where nR is the number of realizations. These 
%      partitions must conform with the corresponding partitions in the 
%      embedded densitye data in the sense that 
%      pairwiseDistance.partition( iR ) must be identical to 
%      denEmbComponent( iC, iR ).partition for all components iC. Pairwise 
%      distances are computed in a block fashion for the elements of the 
%      partition. 
%
%   'kernelDensity': An nlsaKernelDensity object specifying the method and
%      parameters for kernel density estimation. nlsaKernelDensity objects
%      have a property partition which must be identical to the corresponding
%      partition property in denPairwiseDistance. nlsaKernelDensity objects
%      also have a property dimension specifying the intrinsic dimension of
%      the manifold sampled by the data. If this quantity is not known a 
%      priori it can be estimated via one of the dimension estimation 
%      algorithms available in the literature, e.g., [2]. Alternatively, an 
%      esimate for the manifold dimension can be obtained by constructing 
%      a standard nlsaModel for the same data using a diffusion operator 
%      object of class nlsaDiffusionOperator_gl_mb. That class has a method
%      computeOptimalBandwidth which provides an estimate of the manifold 
%      dimension.  
%
%      Two classes, nlsaKernelDensity_fb and nlsaKernelDensity_vb, are 
%      derived from nlsaKernelDensity which perform density estimation using
%      fixed and variable bandwidth kernels, respectively, as described in 
%      [1,3]. Objects of either class estimate the density by selecting an
%      optimal kernel bandwidth parameter from a set of trial bandwidth 
%      parameters. The trial bandwidth parameters are spaced logarithmically, 
%      and are specified using the properties epsilonB and epsilonELim for the
%      base and exponent limits, respectively. nlsaKernelDensity_vb objects
%      additionally have a property kNN specifying the number of nearest
%      neighbors used to compute the variable bandwidth function.   
%  
%   'kernelDensityQ': An array of nlsaComponent objets storing the density data
%      in the original partition if embComponentQ is non-empty. 
%   
%   'embKernelDensity': An [ nC nR ]-sized array of nlsaEmbeddedComponent
%      objects storing the density estimates partioned across the batches
%      of the source data, and (potentially) delay embedded. In particular,
%      the partition property of the 'embKernelDensity' property must be
%      identical to that of 'embComponent'. Moreover, the dimension 
%      property of 'embKernelDensity' must be set to 1, as density is a 
%      scalar quantity. 
%
%      The embedding window length in 'embKernelDensity' can be either 1 
%      or equal to the embedding window length in 'srcComponent'. In the
%      former case, a single density-dependent normalization is applied to the
%      snapshots comprising the delay-embedded source data. In the latter case,
%      the snapshots within the embedding window are scaled individually by
%      the delay-embedded densities.
%
%   'embKernelDensityT': Similar to property 'embComponentT' of the
%      nlsaModel_base parent class, but for the embKernelDensity data.
%      'embKernelDensityT' must be an [ nC 1 ]-sized array of 
%      nlsaEmbeddedComponent_e objects with the same partition as 
%      'embComponentT'. 
%
%   'embKernelDensityQ': As in 'embKernelDensityT', but for the 'embComponentQ'
%      property of the nlsaModel_base parent class.
%
%   Alternatively, the constructor can be called in "template" mode, where 
%   instead of the fully defined objects listed above the arguments supplied
%   by the user only have a set of essential properties defined, and the 
%   remaining properties are filled in automatically. See the class method 
%   parseTemplates for more detais. 
%
%   Below is a summary of selected methods implemented by this class. These
%   methods can be executed in the sequence listed below. The results of each
%   step are written on disk.  
%            
%   - computeDenDelayEmbedding: performs Takens delay embedding on the 
%     out-of-sample data in denComponent; stores the results in
%     denEmbComponent. This step can be skipped if srcComponent and 
%     embComponent are identical to denComponent and denEmbComponent, 
%     respectively.               
%
%   - computeDenVelocity: Computes the phase space velocity for the data in 
%     outEmbComponent. This step can be skipped if srcComponent and 
%     embComponent are identical to denComponent and denEmbComponent, 
%     respectively.    
%
%   - computeDenPairwiseDistances: Computes the pairwise distances between the
%     delay-embedded density data. This mehod supports rudimentary
%     parallelization by splitting the calculation into blocks which can be
%     executed in parallel. 
%
%   - computeDenBandwidthNormalization: Computes the normalization factor used
%     in kernel density estimation with variable-bandwidth kernels. This 
%     step is only needed when density estimation is performed with 
%     nlsaKernelDensity_vb objects.
% 
%   - computeDenKernelDoubleSum: Computes the density estimation kernel sum
%     for trial bandwidth values. This step is needed for automatic 
%     bandwith selection.
%
%   - computeDensity: Computes the kernel density estimates for the model.
%
%   - computeDensityDelayEmbedding: Performs delay embedding for the kernel
%     density estimates; stores the results in embDensity.
% 
%   The following methods can be used to access the results of the calculation:
%
%   - getDensity: Retrieves the kernel density estimates.
%
%   References
%   [1] T. Berry and J. Harlim (2015), "Variable bandwidth diffusion kernels",
%       Appl. Comput. Harmon. Anal., doi:10.1016/j.acha.2015.01.001
%   [2] A. V. Little, J. Lee, Y.-M. Jung, and M. Maggioni (2009), "Estimation 
%       of intrinsicd imensionality of samples from noisylow-dimensional
%       manifolds in high dimensions with multiscale SVD, Proc. 15th IEEE/SP
%       Workshop on Statistical Signal Processing, 85--88, 
%       doi:10.1109/SSP.2009.5278634
%   [3] T. Berry, D. Giannakis, and J. Harlim (2015), "Nonparametric 
%       forecasting of low-dimensional dynamical systems", Phys. Rev. E, 91, 
%       032915, doi:10.1103/PhysRevE.91.032915
%
%   Contact: dimitris@cims.nyu.edu
%      
%   Modified 2020/03/19

    %% PROPERTIES

    properties
        denComponent       = nlsaComponent();
        denEmbComponent    = nlsaEmbeddedComponent_e();
        denEmbComponentT   = nlsaEmbeddedComponent_e.empty;
        denEmbComponentQ   = nlsaEmbeddedComponent_e.empty;
        denPDistance       = nlsaPairwiseDistance();
        density            = nlsaKernelDensity_fb();
        densityQ           = nlsaComponent.empty; 
        embDensity         = nlsaEmbeddedComponent_e(); 
        embDensityT        = nlsaEmbeddedComponent_e.empty;
        embDensityQ        = nlsaEmbeddedComponent_e.empty;
    end
    
    methods
        
        %% CLASS CONSTRUCTOR
        function obj = nlsaModel_den( varargin )
            
            msgId = 'nlsa:nlsaModel_den:';
            
            % Check if constructor is called in "template" mode and parse
            % templates if needed
            if ifTemplate( 'nlsaModel_den', varargin{ : } )
                varargin = nlsaModel_den.parseTemplates( varargin{ : } );
            end
            nargin   = numel( varargin );
            ifParentArg = true( 1, nargin ); 

            % Parse input arguments 
            iDenComponent     = [];
            iDenEmbComponent  = [];
            iDenEmbComponentT = [];
            iDenEmbComponentQ = [];
            iDenPDistance     = [];
            iDensity          = [];
            iDensityQ         = [];
            iEmbDensity       = [];
            iEmbDensityT      = [];
            iEmbDensityQ      = [];
            
            for i = 1 : 2 : nargin
                switch varargin{ i }                    
                    case 'denComponent'
                        iDenComponent = i + 1;
                        ifParentArg( [ i i + 1 ] )  = false;
                    case 'denEmbComponent'
                        iDenEmbComponent = i + 1;                        
                        ifParentArg( [ i i + 1 ] )  = false;
                    case 'denEmbComponentT'
                        iDenEmbComponentT = i + 1;                        
                        ifParentArg( [ i i + 1 ] )  = false;
                    case 'denEmbComponentQ'
                        iDenEmbComponentQ = i + 1;                        
                        ifParentArg( [ i i + 1 ] )  = false;
                    case 'denPairwiseDistance'
                        iDenPDistance = i + 1;
                        ifParentArg( [ i i + 1 ] ) = false;
                    case 'kernelDensity'
                        iDensity = i + 1;
                        ifParentArg( [ i i + 1 ] ) = false;
                    case 'kernelDensityQ'
                        iDensityQ = i + 1;
                        ifParentArg( [ i i + 1 ] ) = false;
                    case 'embKernelDensity'
                        iEmbDensity = i + 1;
                        ifParentArg( [ i i + 1 ] ) = false;
                    case 'embKernelDensityQ'
                        iEmbDensityQ = i + 1;
                        ifParentArg( [ i i + 1 ] ) = false;
                    case 'embKernelDensityT'
                        iEmbDensityT = i + 1;
                        ifParentArg( [ i i + 1 ] ) = false;
                end
            end

            obj = obj@nlsaModel( varargin{ ifParentArg } ); 

            
            % Source data for density estimation 
            if ~isempty( iDenComponent )
                if ~isa( varargin{ iDenComponent }, 'nlsaComponent' )
                    error( [ msgId 'invalidDensity' ], ...
                           'Density data must be specified as an array of nlsaComponent objects.' )
                end
            
                [ ifC, Test1, Test2 ] = isCompatible( obj.srcComponent, ...
                                                      varargin{ iDenComponent }, ...
                                                      'testComponents', false, ...
                                                      'testSamples', true );
                if ~ifC
                    disp( Test1 )
                    disp( Test2 )
                    error( [ msgId 'incompatibleDensity' ], ...
                           'Incompatible Density component array' )
                end
                obj.denComponent = varargin{ iDenComponent }; 
            else
                obj.denComponent = obj.srcComponent;
            end
            nCD = size( obj.denComponent, 1 );

            % Density embedded components
            % Check input array class and size
            
            if ~isempty( iDenEmbComponent )
                if ~isa( varargin{ iDenEmbComponent }, ...
                         'nlsaEmbeddedComponent' )
                    error( [ msgId 'invalidDensityEmb' ], ...
                           'Embedded density data must be specified as an array of nlsaEmbeddedComponent objects.' )
                end

                % Check constistency of data space dimension, embedding indices
                %, and number of samples
                [ ifC, Test1, Test2 ] = isCompatible( ...
                  varargin{ iDenEmbComponent }, ...
                  obj.denComponent, ...
                  'testComponents', true, ...
                  'testSamples', false );
                if ~ifC
                    msgStr = 'Incompatible Density embedded component array';
                    disp( Test1 )
                    disp( Test2 )
                    error( [ msgId 'incompatibleDenEmb' ], msgStr )
                end
                obj.denEmbComponent = varargin{ iDenEmbComponent };
            else
                obj.denEmbComponent = nlsaEmbeddedComponent_e( ...
                  obj.denComponent );
            end
            nSETot = getNEmbSample( obj );                    
            partition = getEmbPartition( obj ); 

            % Embedded density components (test)
            if ~isempty( iDenEmbComponentT )
                if ~isa( varargin{ iDenEmbComponentT }, ...
                         'nlsaEmbeddedComponent' ) ...
                    || ~iscolumn( varargin{ iDenEmbComponentT } )

                    error( [ msgId 'invalidEmbT' ], ...
                           'Embedded test data must be specified as a column vector of nlsaEmbeddedComponent objects.' )
                end
                nCET = size( varargin{ iDenEmbComponentT }, 1 ); 
                if nCET ~= nCD 
                    msgStr = sprintf( [ 'Invalid row dimension of embedded component array: \n' ...
                                        'Expecting [%i] \n' ...
                                        'Received  [%i]' ], ...
                                      nCD, nCET );
                    error( [ msgId 'invalidEmbT' ], msgStr )
                end
                obj.denEmbComponentT = varargin{ iDenEmbComponentT };
                partitionT = getPartition( obj.denEmbComponentT( 1 ) );
                if ~isFiner( partition, partitionT )
                    error( 'Test partition for the embedded density data must be a refinement of the partition for the embedded density data' )
                end
            else
                % If obj.denEmbComponentT is empty, set partitionT to empty;
                % this will be used later on to construuct 
                % obj.denPairwiseDistance
                partitionT = nlsaPartition.empty;
            end

            % Embedded density components (query)
            if ~isempty( iDenEmbComponentQ )
                if ~isa( varargin{ iDenEmbComponentQ }, ...
                         'nlsaEmbeddedComponent' ) ...
                    || ~iscolumn( varargin{ iDenEmbComponentQ } )

                    error( [ msgId 'invalidEmbQ' ], ...
                           'Embedded query data must be specified as a column vector of nlsaEmbeddedComponent objects.' )
                end
                nCEQ = size( varargin{ iDenEmbComponentQ }, 1 ); 
                if nCEQ ~= nCD 
                    msgStr = sprintf( [ 'Invalid row dimension of embedded component array: \n' ...
                                        'Expecting [%i] \n' ...
                                        'Received  [%i]' ], ...
                                      nCD, nCEQ );
                    error( [ msgId 'invalidEmbQ' ], msgStr )
                end
                obj.denEmbComponentQ = varargin{ iDenEmbComponentQ };
                partitionQ = getPartition( obj.denEmbComponentQ( 1 ) );
                if ~isFiner( partition, partitionQ )
                    error( 'Query partition for the embedded density data must be a refinement of the partition for the embedded density data' )
                end
            else
                % If obj.denEmbComponentQ is empty, set partitionQ to the
                % partition in obj.denEmbComponent; this will be used later on 
                % to construuct obj.denPairwiseDistance
                partitionQ = getPartition( obj.denEmbComponent( 1, : ) );
            end

            % Density pairwise distance
            if ~isempty( iDenPDistance )
                if ~isa( varargin{ iDenPDistance }, 'nlsaPairwiseDistance' ) 
                       error( [ msgId 'invalidDensityPDist' ], ...
                              'The density pairwise distance property must be an nlsaPairwiseDistance object' )
                end
                if ~( isscalar( varargin{ iDenPDistance } ) ...
                      || numel( varargin{ iDenPDistance } ) == ...
                        size( obj.denComponent, 1 ) )
                      error( 'The density pairwise distance property must be scalar or a vector of size equal to the number of source components' )
                end
                if any ( getNNeighbors( varargin{ iDenPDistance } ) ...
                  > getNTotalSample( varargin{ iDenPDistance } ) ) 
                    error( 'The number of nearest neighbors cannot exceed the number of density samples' )
                end
                obj.denPDistance = varargin{ iDenPDistance };
            else
                obj.denPDistance = nlsaPairwiseDistance( ...
                                'partition', partitionQ, ...
                                'partitionT', partitionT, ...
                                'nearestNeighbors', round( nSETot / 10 ) );
            end

            % Kernel density
            if ~isempty( iDensity )
                if ~isa( varargin{ iDensity }, 'nlsaKernelDensity' ) ...
                    || any( size( varargin{ iDensity } ) ~= size( varargin{ iDenPDistance } ) )
                       error( [ msgId 'invalidDensity' ], ...
                              'The kernelDensity  property must be specified as a nlsaKernelDensity object of size equal to the density pairwise distance' )
                end
                obj.density = varargin{ iDensity };
            else
                obj.density = nlsaKernelDensity_fb( ...
                                 'partition', partitionQ );
            end

            % Kernel density (query)
            if ~isempty( iDensityQ )
                if isempty( obj.denEmbComponentQ )
                    error( 'densityQ property can only be set if denEmbComponentQ is set.' )
                end
                if size( varargin{ iDensityQ }, 1 ) ~= size( obj.denPDistance )
                   error( 'Incompatible number of components in densityQ property' )
               end 
                if any( getDataSpaceDimension( varargin{ iDensityQ }( :, 1 ) ) ~= 1 ) 
                    error( 'Data space dimension of densityQ property must be equal to 1.' )
                end
                if any( ~isequal( partition, getPartition( ...
                    varargin{ iDensityQ }( 1, : ) ) ) )
                   error( 'Incompatible partition for densityQ property.' )
               end
               obj.densityQ = varargin{ iDensityQ };
            elseif ~isempty( obj.denEmbComponentQ )
               % This case needs implementation
               obj.densityQ = nlsaComponent();
            end
            
            % Delay-embedded kernel density
            if ~isempty( iEmbDensity )
                if ~isa( varargin{ iEmbDensity }, 'nlsaEmbeddedComponent' ) ...
                  || size( varargin{ iEmbDensity }, 1 ) ~= ...
                          numel( obj.denPDistance ) 
                    error( 'The embedded density must be an array of nlsaEmbeddedComponent objects with number of row equal to the number of elements of the denPDistance array' )
                end
                if ~isCompatible( varargin{ iEmbDensity }, ...
                                  getEmbComponent( obj ), ...
                                  'testComponents', false )
                    error( 'Incompatible embedded density array' )
                end
                if any( getDimension( varargin{ iEmbDensity } ) ~= 1 )
                    error( 'Invalid embedded density dimension' )
                end
                obj.embDensity = varargin{ iEmbDensity };
            else
                obj.embDensity = nlsaEmbeddedComponent_e();
            end
            
            % Embedded density (test)
            if ~isempty( iEmbDensityT )
                if ~isa( varargin{ iEmbDensityT }, ...
                         'nlsaEmbeddedComponent_e' ) ...
                    || ~iscolumn( varargin{ iEmbDensityT } ) 
                    error( [ msgId 'invalidEmbT' ], ...
                           'Embedded density data (test) must be specified as a column vector of nlsaEmbeddedComponent_e objects.' )
                end
                nCET = size( varargin{ iEmbDensityT }, 1 ); 
                if nCET ~= nCD 
                    msgStr = sprintf( [ 'Invalid row dimension of embedded density component array: \n' ...
                                        'Expecting [%i] \n' ...
                                        'Received  [%i]' ], ...
                                      nC, nCET );
                    error( [ msgId 'invalidEmbT' ], msgStr )
                end
                obj.embDensityT = varargin{ iEmbDensityT };
                if ~isequal( obj.embComponentT( 1 ).partition, ...
                             obj.embDensityT( 1 ).partition )
                    error( 'Test partition for the embedded density data must be equal to the source partition of the embedded data' )
                end
            end

            % Embedded density (query)
            if ~isempty( iEmbDensityQ )
                if ~isa( varargin{ iEmbDensityQ }, ...
                         'nlsaEmbeddedComponent_e' ) ...
                    || ~iscolumn( varargin{ iEmbDensityQ } ) 
                    error( [ msgId 'invalidEmbQ' ], ...
                           'Embedded density data (test) must be specified as a column vector of nlsaEmbeddedComponent_e objects.' )
                end
                nCEQ = size( varargin{ iEmbDensityQ }, 1 ); 
                if nCEQ ~= nCD 
                    msgStr = sprintf( [ 'Invalid row dimension of embedded density component array: \n' ...
                                        'Expecting [%i] \n' ...
                                        'Received  [%i]' ], ...
                                      nC, nCEQ );
                    error( [ msgId 'invalidEmbQ' ], msgStr )
                end
                obj.embDensityQ = varargin{ iEmbDensityQ };
                if ~isequal( obj.embComponentQ( 1 ).partition, ...
                             obj.embDensityQ( 1 ).partition )
                    error( 'Query partition for the embedded density data must be equal to the source partition of the embedded data' )
                end
            end
        end
    end
         
    methods( Static )
   
        %% LISTCONSTRUCTORPROPERTIES  List property names for class constructor 
        function pNames = listConstructorProperties
            pNames = nlsaModel.listConstructorProperties;
            pNames = [ pNames ...
                       { 'denComponent' ...
                         'denEmbComponent' ...
                         'denEmbComponentT' ...
                         'denEmbComponentQ' ...
                         'denPairwiseDistance' ...
                         'kernelDensity' ...
                         'kernelDensityQ' ...
                         'embKernelDensity' ...
                         'embKernelDensityT' ...
                         'embKernelDensityQ' } ];
        end

        %% LISTPARSERPROPERTIES List property names for class constructor parser
        % This should only return parser properties which are not constructor
        % properties
        function pNames = listParserProperties
            pNames = nlsaModel.listParserProperties;
            pNames = [ pNames ...
                       { 'densityComponentName' ...
                         'densityRealizationName' ...
                         'denEmbeddingTemplate' ... 
                         'denEmbeddingPartition' ...
                         'denPairwiseDistanceTemplate' ...
                         'kernelDensityTemplate' ...
                         'densityEmbeddingTemplate' } ];
        end
            
        %% PARSETEMPLATES  Template parser
        propNameVal = parseTemplates( varargin );
    end    
end
