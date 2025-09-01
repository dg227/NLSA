classdef nlsaModel_den_ose < nlsaModel_den
%NLSAMODEL_OSE   Class definition and constructor of NLSA model with
%  kernel density estimation and out-of-sample extension (OSE).
%
%  nlsaModel_den_ose builds on the nlsaModel_den parent class to 
%  implement kernel density estimation and OSE of the diffusion eigenfunctions
%  and the associated reconstructed patterns for previously unseen 
%  (out-of-sample) data. In particular, nlsaModel_ose 
%  implements methods for the following OSE operations:
%
%  THIS DESCRIPTION NEEDS TO BE UPDATED
%
%  (i)   Delay embedding of the out-of-sample data and the corresponding phase
%        space velocities.
%
%  (ii)  Calculation of the pairwise distances between the source (in-sample) 
%        data and the out-of-sample data. 
%
%  (iii) Construction of a kernel diffusion operator for OSE.
%
%  (iv)  OSE of the delay-embedded target data using the operator in (ii)
%        directly or via the the Nystrom method [1]. 
%
%  (v)   Recontruction of the out-of-sample data in physical 
%        (non-delay-embedded) space. 
%
%  The class constructor arguments are passed as property name-property value
%  pairs using the syntax
%
%  model = nlsaModel( propName1, propVal1, propName2, propVal2, ... ).
%
%  In addition to the of the nlsaModel_base parent class, the following 
%  properties can be specified:
%
%   'outComponent': An [ nC nRO ]-sized array of nlsaComponent objects 
%      specifying the out-of-sample source data. nC is the number of 
%      components (physical variables) and nRO the number of realizations
%      (ensemble members) in the out-of-sample data. nC must be equal to the
%      number of components in the in-sample source data as specified in
%      'srcComponent'.
%
%   'outTrgComponent': An [ nCT nRO ]-sized array of nlsaComponent objects
%      specifying the out-of-sample target data. nCT is the number of target
%      components, and must be equal to the number of target components in the
%      in-sample target data as specified in 'trgComponent'. This property
%      can be left empty if there are no out-of-sample target data. 
%
%   'outTime': An [ 1 nRO ]-sized cell array of vectors specifying the time
%      stamps of each sample in the out-of-sample data. The number of elements
%      in outTime{ iR } must be equal to the number of samples in 
%      outComponent( :, iR ). If 'outTime' is not specified it is set to the
%      default values outTime{ iR } = 1 : nSO( iR ), where nSO( iR ) is the 
%      number of samples in the iR-th realization of the out-of-sample data.
% 
%   'outEmbComponent': An [ nC nRO ]-sized array of nlsaEmbeddedComponent
%      objects storing Takens delay-embedded data associacted with
%      'outComponent'. See the documentation of the nlsaModel_base class 
%      for additional information on nlsaEmbeddedComponent objects.
%
%   'outTrgEmbComponent': As in 'outEmbComponent', but for the target data.
%
%   'osePairwiseDistance': An nlsaPairwiseDistance object specifying
%      the method and parameters (e.g., number of nearest neighbors) used 
%      to compute the pairwise distances between the in-sample and 
%      out-of-sample delay-embedded data. 'osePairwiseDistance' operates on the
%      data specified in properties 'embComponent' and 'outEmbComponent' of 
%      the model. 
%     
%      nlsaPairwiseDistance objects have a property partition which specifies
%      the partitioning of the in-sample data. This property must be set to a
%      vector of nlsaPartition objects of size [ 1 nR ], where nR is the number
%      of realizations in the in-sample data. These partitions must conform 
%      with the corresponding partitions in the embedded source data in the
%      sense that pairwiseDistance.partition( iR ) must be identical to 
%      embComponent( iC, iR ).partition for all components iC. 
%     
%      nlsaPairwiseDistance objects also have a property partitionT specifying
%      the partitioning of the out-of-sample data. This property must be
%      set to a vector of nlsaPartition objects of size [ 1 nRO ]. These
%      partitions must conform with the corresponding partitions in the
%      embedded source data in the sense that pairwiseDistance.partitionT( iR )
%      must be identical to outEmbComponent( iC, iR ).partition for all
%      components iC. 
%
%      See the documentation of the nlsaModel class for additional information
%      on nlsaPairwiseDistance objects. 
%
%   'oseDiffOp': An nlsaDiffusionOperator_ose object implementing the kernel 
%      diffusion operator for out-of-sample extension. The
%      nlsaDiffusionOperator_ose class is derived from the 
%      nlsaDiffusionOperator_batch class, which is described in the 
%      documentation of the nlsaModel class. In particular, 
%      nlsaDiffusionOperator_ose objects have the properties partition and 
%      partitionT which must be identical to the corresponding properties 
%      specified in 'osePairwiseDistance'.
%
%   'oseEmbComponent': An [ nCT nRO ]-sized array of nlsaEmbeddedComponent_ose
%      objects storing the out-of-sample extension of the target data in 
%      Takens delay-embedding space. The nlsaEmbeddedComponent_ose class is
%      derived from the nlsaEmbeddedComponent_xi_e class. As a result, data
%      in nlsaEmbeddedComponent_ose objects are always in 'evector' format.
%      
%      'oseEmbComponent' operates on the data in 'trgEmbComponent' using the
%      kernel operator in 'oseDiffOp'. Out-of-sample extension with the
%      nlsaEmbeddedComponent_ose class is implemented via averaging through
%      the kernel operator in 'oseDiffOp'. In addition, a derived class
%      nlsaEmbeddedComponent_ose_n is provided that carries out out-of-sample
%      extension via the Nystrom method [1]. nlsaEmbeddedComponent_ose_n
%      objects have a property idxPhi which is a vector of integers specifying
%      the eigenfunctions of 'oseDiffOop' used for out-of-sample extension.
% 
%   'oseRecComponent': An [ nCT nR ]-sized array of nlsaComponent_rec objects 
%      implementing the reconstruction of the data in 'oseEmbComponent' in the
%      physical data space. The nlsaComponent_rec class is a child 
%      of the nlsaComponent class. In particular, it has a dimension property
%      nD which must be compatible with the dimension of the target data, i.e., 
%      oseRecComponent( iCT, iR ).nD must be equal to 
%      trgComponent( iCT,iR ).nD. nlsaComponent_rec objects also have a
%      partition property which must be set to an nlsaPartition object.
%      The number of samples in recComponent( iCT, iR ).partition must not
%      exceed the number of samples in the delay embedded data, 
%      oseEmbComponent( iCT, iR ).partition, plus nE( iCT ) - 1, where 
%      nE( iCT ) is the number of delays for target component iCT.
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
%   - computeOutDelayEmbedding: performs Takens delay embedding on the 
%     out-of-sample data in outComponent; stores the results in outEmbComponent.
%
%   - computeOutVelocity: Computes the phase space velocity for the data in 
%     outEmbComponent.
%
%   - computeOsePairwiseDistances: Computes the pairwise distances between the
%     in-sample and out-of-sample delay-embedded data. This mehod supports
%     rudimentary parallelization by splitting the calculation into blocks
%     which can be executed in parallel. 
%
%   - computeOseKernelNormalization, computeOseKernelDegree: Computes the 
%     kernel normalization and kernel degree for out-of-sample extension.
%
%   - computeOseDiffusionOperator: Computes the diffusion operator for OSE from 
%     the pairwise distance data in 'osePairwiseDistance'. 
%
%   - computeOseDiffusionEigenfunctions: Performs out-of-sample extension of
%     the diffusion eigenfunctions computed from the in-sample data.  
%
%   - computeOseEmbData: Computes the out-of-sample extension of the target
%     data in delay-embedding space.
%
%   - computeOseReconstruction: Performs reconstruction (projection) of the 
%     OSE data in delay-embedding space in the physical data space.
%
%   The following methods can be used to access the results of the calculation:
%
%   - getOseDiffusionEigenfunctions: Retrieves the OSE diffusion
%     eigenfunctions. 
%
%   - getOseReconstructedData: Retrieves the OSE reconstructed data.
%
%   References
%   [1] R. R. Coifman and S. Lafon (2006), "Geometric harmonics: A novel tool
%       for multiscale out-of-sample extension of empirical functions", Appl. 
%       Comput. Harmon. Anal., 21, 31-52, doi:10.1016/j.acha.2005.07.005
%
%  Contact: dimitris@cims.nyu.edu
% 
%  Modified 2020/08/02

    %% PROPERTIES
    properties
        outComponent       = nlsaComponent();
        outEmbComponent    = nlsaEmbeddedComponent_e();
        outTrgComponent    = nlsaComponent.empty();
        outTrgEmbComponent = nlsaEmbeddedComponent_e.empty();
        outDenComponent    = nlsaComponent();
        outDenEmbComponent = nlsaEmbeddedComponent_e();
        outTime            = { 1 };
        outTimeFormat      = '';
        oseDenPDistance    = nlsaPairwiseDistance();
        oseDensity         = nlsaKernelDensity_ose_fb();
        oseEmbDensity      = nlsaEmbeddedComponent_e();
        osePDistance       = nlsaPairwiseDistance();
        oseDiffOp          = nlsaDiffusionOperator_ose();
        oseEmbComponent    = nlsaEmbeddedComponent_ose();
        oseRecComponent    = nlsaComponent_rec();
    end
    
    methods
        
        %% CLASS CONSTRUCTOR
        function obj = nlsaModel_den_ose( varargin )
            
            msgId = 'nlsa:nlsaModel_den_ose:';
            
            % Check if constructor is called in "template" mode and parse
            % templates if needed
            if ifTemplate( 'nlsaModel_den_ose', varargin{ : } )
                varargin = nlsaModel_den_ose.parseTemplates( varargin{ : } );
            end
            nargin   = numel( varargin );
            ifParentArg = true( 1, nargin ); 

            % Parse input arguments 
            iOutComponent       = [];
            iOutEmbComponent    = [];
            iOutTrgComponent    = [];
            iOutTrgEmbComponent = [];
            iOutDenComponent    = [];
            iOutDenEmbComponent = [];
            iOutTime            = [];
            iOutTimeFormat      = [];
            iOutDenComponent    = [];
            iOutDenEmbComponent = [];
            iOseDenPDistance    = [];
            iOseDensity         = [];
            iOseEmbDensity      = [];
            iOsePDistance       = [];
            iOseDiffOp          = [];
            iOseEmbComponent    = [];
            iOseRecComponent    = [];

            for i = 1 : 2 : nargin
                switch varargin{ i }                    
                    case 'outComponent'
                        iOutComponent = i + 1;
                        ifParentArg( [ i i + 1 ] )  = false;
                    case 'outEmbComponent'
                        iOutEmbComponent = i + 1;                        
                        ifParentArg( [ i i + 1 ] )  = false;
                    case 'outTrgComponent'
                        iOutTrgComponent = i + 1;
                        ifParentArg( [ i i + 1 ] )  = false;
                    case 'outTrgEmbComponent'
                        iOutTrgEmbComponent = i + 1;                        
                        ifParentArg( [ i i + 1 ] )  = false;
                    case 'outDenComponent'
                        iOutDenComponent = i + 1;
                        ifParentArg( [ i i + 1 ] )  = false;
                    case 'outDenEmbComponent'
                        iOutDenEmbComponent = i + 1;                        
                        ifParentArg( [ i i + 1 ] )  = false;
                    case 'outTime'
                        iOutTime = i + 1;
                        ifParentArg( [ i i + 1 ] ) = false;
                    case 'outTimeFormat'
                        iOutTimeFormat = i + 1;
                        ifParentArg( [ i i + 1 ] ) = false;
                    case 'oseDenPairwiseDistance'
                        iOseDenPDistance = i + 1;
                        ifParentArg( [ i i + 1 ] ) = false;
                    case 'oseKernelDensity'
                        iOseDensity = i + 1;
                        ifParentArg( [ i i + 1 ] ) = false;
                    case 'oseEmbKernelDensity'
                        iOseEmbDensity = i + 1;
                        ifParentArg( [ i i + 1 ] ) = false;
                    case 'osePairwiseDistance'
                        iOsePDistance = i + 1;
                        ifParentArg( [ i i + 1 ] ) = false;
                    case 'oseDiffusionOperator'
                        iOseDiffOp = i + 1;
                        ifParentArg( [ i i + 1 ] ) = false;
                    case 'oseEmbComponent'
                        iOseEmbComponent = i + 1;
                        ifParentArg( [ i i + 1 ] ) = false;
                    case 'oseRecComponent'
                        iOseRecComponent = i + 1;
                        ifParentArg( [ i i + 1 ] ) = false;
                end
            end

            obj = obj@nlsaModel_den( varargin{ ifParentArg } ); 

            % Query partitions of in-sample data become test partitions for
            % OSE
            partitionDT = getDenEmbPartitionQ( obj );
            partitionT  = getEmbPartitionQ( obj );
            pDistance = getPairwiseDistance( obj );
            nNT = getNNeighbors( pDistance );

            % OS components 
            if ~isempty( iOutComponent )
                if ~isa( varargin{ iOutComponent }, 'nlsaComponent' )
                    error( [ msgId 'invalidOSE' ], ...
                           'OSE data must be specified as an array of nlsaComponent objects.' )
                end
            else
                error( [ msgId 'emptyOSE' ], 'Unassigned OSE components' )
            end
            
            [ ifC, Test1, Test2 ] = isCompatible( ...
                obj.srcComponent, ...
                varargin{ iOutComponent }, ...
                'testComponents', true, ...
                'testSamples', false );

            if ~ifC
                disp( Test1 )
                disp( Test2 )
                error( [ msgId 'incompatibleOSE' ], ...
                       'Incompatible OSE component array' )
            end
            obj.outComponent = varargin{ iOutComponent };     
            [ nC, nR ] = size( obj.outComponent );
            nSR        = getNSample( obj.outComponent( 1, : ) );   % number of samples in each realization

            % OS target components
            if ~isempty( iOutTrgComponent )
                if ~isa( varargin{ iOutTrgComponent }, 'nlsaComponent' )
                    error( [ msgId 'invalidOSE' ], ...
                           'OSE target data must be specified as an array of nlsaComponent objects.' )
                end
            
                [ ifC, Test1, Test2 ] = isCompatible( ...
                    obj.trgComponent, ...
                    varargin{ iOutTrgComponent }, ...
                    'testComponents', true, ...
                    'testSamples', false );
                        
                if ~ifC
                    disp( Test1 )
                    disp( Test2 )
                    error( [ msgId 'incompatibleOSE' ], ...
                           'Incompatible OSE component array' )
                end
                obj.outTrgComponent = varargin{ iOutTrgComponent };     
                nCT = size( obj.outTrgComponent, 1 );
            end


            % OS components for density estimation
            if ~isempty( iOutDenComponent )
                if ~isa( varargin{ iOutDenComponent }, 'nlsaComponent' )
                    error( [ msgId 'invalidDENOSE' ], ...
                           'OSE data for density estimation must be specified as an array of nlsaComponent objects.' )
                end
            
                [ ifC, Test1, Test2 ] = isCompatible( ...
                    obj.srcComponent, ...
                    varargin{ iOutComponent }, ...
                    'testComponents', false, ...
                    'testSamples', true );

                if ~ifC
                    disp( Test1 )
                    disp( Test2 )
                    error( [ msgId 'incompatibleDenOSE' ], ...
                           'Incompatible density OSE component array' )
                end
                obj.outDenComponent = varargin{ iOutDenComponent };     
            else
                obj.outDenComponent = obj.outComponent;
            end


            % Time for OS data
            if ~isempty( iOutTime )
                if isvector( varargin{ iOutTime } ) && ~iscell( varargin{ iOutTime } )
                    obj.outTime = cell( 1, nR );
                    for iR = 1 : nR
                        obj.outTime{ iR } = varargin{ iOutTime };
                    end
                elseif   isvector( varargin{ iOutTime } ) && iscell( varargin{ iOutTime } ) ...
                      && numel( varargin{ iOutTime } ) == nR
                    obj.outTime = varargin{ iOutTime };
                else
                    error( [ msgId 'invalidTime' ], ...
                               'Time data must be either a vector or a cell vector of size nR.' )
                end
                if size( obj.outTime, 1 ) > 1
                    obj.outTime = obj.outTime';
                end
                for iR = 1 : nR
                    if ~isvector( obj.outTime{ iR } ) 
                        msgStr = sprintf( [ 'Invalid timestamp array for realization %i: \n' ... 
                                            'Expecting vector of size %i \n' ...
                                            'Received  array of size [%i] \n' ], ...  
                                            iR, nSR( iR ), size( obj.outTime{ iR } ) );  
                        error( [ msgId 'invalidOSETimestamps' ], msgStr )
                    end
                    
                    if numel( obj.outTime{ iR } ) ~= nSR( iR )
                        msgStr = sprintf( [ 'Invalid number of timestamps for realization %i: \n' ... 
                                            'Expecting %i \n' ...
                                            'Received  %i \n' ], ...  
                                            iR, nSR( iR ), numel( obj.outTime{ iR } ) );  
                        error( [ msgId 'invalidOSETimestamps' ], msgStr )
                    end
                    if ~ismonotonic( obj.outTime{ iR }, [], 'increasing' )
                        msgStr = sprintf( [ 'Invalid timestamps for realization %i: \n' ... 
                                            'Time values must be monotonically increasing.\n' ], ...
                                          iR );
                        error( [ msg 'invalidOSETimestamps' ], msgStr )
                    end
                    if size( obj.outTime{ iR }, 1 ) > 1
                        obj.outTime{ iR } = obj.outTime{ iR }';
                    end
                end
            else % set to default timestamps if no caller input
                obj.outTime = cell( 1, nR );
                for iR = 1 : nR
                    obj.outTime{ iR } = 1 : nSR( iR );
                end
            end
           
            % OS time format
            if ~isempty( iOutTimeFormat )
                if ~ischar( varargin{ iOutTimeFormat } )
                   error( [ msgId 'outTimeFormat' ], 'Invalid time format for OSE data' )
                end
                obj.outTimeFormat = varargin{ iOutTimeFormat };
            end

            % OS embedded components
            % Check input array class and size
            
            if ~isempty( iOutEmbComponent )
                if ~isa( varargin{ iOutEmbComponent }, 'nlsaEmbeddedComponent' )
                    error( [ msgId 'invalidOSEEmb' ], ...
                           'Embedded OSE data must be specified as an array of nlsaEmbeddedComponent objects.' )
                end
            
                % Check constistency of data space dimension, embedding 
                % indices, and number of samples
                [ ifC, Test1, Test2 ] = isCompatible( ...
                    varargin{ iOutEmbComponent }, ...
                    obj.embComponent, ...
                    'testComponents', true, ...
                    'testSamples', false );
                if ~ifC
                    msgStr = 'Incompatible OSE embedded component array';
                    disp( Test1 )
                    disp( Test2 )
                    error( [ msgId 'incompatibleOutEmb' ], msgStr )
                end

                [ ifC, Test1, Test2 ] = isCompatible( ...
                    varargin{ iOutEmbComponent }, ...
                    obj.outComponent, ...
                    'testComponents', true, ...
                    'testSamples', false );

                if ~ifC
                    msgStr = 'Incompatible OSE embedded component array';
                    disp( Test1 )
                    disp( Test2 )
                    error( [ msgId 'incompatibleOutEmb' ], msgStr )
                end

                obj.outEmbComponent = varargin{ iOutEmbComponent };
            else
                error( 'Unspecified OSE components' )
            end
            nSETot = getNOutEmbSample( obj );                    
            partition = getOutEmbPartition( obj ); 
            
            % OS target embedded components
            % Check input array class and size
            
            if ~isempty( iOutTrgEmbComponent )
                if isempty( iOutTrgComponent )
                    error( 'Empty OSE target component' )
                end
                if ~isa( varargin{ iOutEmbComponent }, 'nlsaEmbeddedComponent' )
                    error( [ msgId 'invalidOSEEmb' ], ...
                           'Embedded OSE data must be specified as an array of nlsaEmbeddedComponent objects.' )
                end
            
                % Check constistency of data space dimension, embedding indices, and 
                % number of samples
                [ ifC, Test1, Test2 ] = isCompatible( ...
                    varargin{ iOutTrgEmbComponent }, ...
                    obj.trgEmbComponent, ...
                    'testComponents', true, ...
                    'testSamples',    false );
                if ~ifC
                    msgStr = 'Incompatible OSE target embedded component array';
                    disp( Test1 )
                    disp( Test2 )
                    error( [ msgId 'incompatibleOutTrgEmb' ], msgStr )
                end

                [ ifC, Test1, Test2 ] = isCompatible( ...
                    varargin{ iOutTrgEmbComponent }, ...
                    obj.outTrgComponent, ...
                    'testComponents', true, ...
                    'testSamples',    false );

                if ~ifC
                    msgStr = 'Incompatible OSE embedded component array';
                    disp( Test1 )
                    disp( Test2 )
                    error( [ msgId 'incompatibleOutTrgEmb' ], msgStr )
                end

                obj.outTrgEmbComponent = varargin{ iOutTrgEmbComponent };
            elseif ~isempty( iOutTrgEmbComponent )
                error( 'Unspecified target OSE embedded components' )
            end

            % OS density  embedded components
            % Check input array class and size
            
            if ~isempty( iOutDenEmbComponent )
                if ~isa( varargin{ iOutDenEmbComponent }, 'nlsaEmbeddedComponent' )
                    error( [ msgId 'invalidOSEDenEmb' ], ...
                           'Embedded density OSE data must be specified as an array of nlsaEmbeddedComponent objects.' )
                end
            
                % Check constistency of data space dimension, embedding indices, and 
                [ ifC, Test1, Test2 ] = isCompatible( varargin{ iOutDenEmbComponent }, ...
                                                      obj.denEmbComponent, ...
                                                      'testComponents', true, ...
                                                      'testSamples', false );
                if ~ifC
                    msgStr = 'Incompatible density OSE embedded component array';
                    disp( Test1 )
                    disp( Test2 )
                    error( [ msgId 'incompatibleOutDenEmb' ], msgStr )
                end
                obj.outDenEmbComponent = varargin{ iOutDenEmbComponent };
            else
                obj.outDenEmbComponent = nlsaEmbeddedComponent_e( obj.denComponent );
            end
            nSEDTot = getNOutDenEmbSample( obj );                    
            partitionD = getOutDenEmbPartition( obj ); 

            % Density OSE pairwise distance
            if ~isempty( iOseDenPDistance )
                if ~isa( varargin{ iOseDenPDistance }, 'nlsaPairwiseDistance' ) 
                       error( [ msgId 'invalidOSEDenPDist' ], ...
                              'The OSE density pairwise distance property must be an nlsaPairwiseDistance object' )
                end

                if ~( isscalar( varargin{ iOseDenPDistance } ) ...
                      || numel( varargin{ iOseDenPDistance } ) == nC )
                      error( 'The density pairwise distance property must be scalar or a vector of size equal to the number of source components' )
                end
                if getNNeighbors( varargin{ iOseDenPDistance } ) ...
                  > getNTotalSampleTest( varargin{ iOseDenPDistance } ) 
                    error( 'The number of nearest neighbors cannot exceed the number of test density samples' )
                end
                obj.oseDenPDistance = varargin{ iOseDenPDistance };
            else
                obj.oseDenPDistance = nlsaPairwiseDistance( ...
                                    'partition', partitionD, ...
                                    'partitionT', partitionDT, ...
                                    'nearestNeighbors', round( nSETot / 10 ) ); 
            end
            nNO = getNNeighbors( obj.oseDenPDistance );
 

            % OSE kernel density
            if ~isempty( iOseDensity )
                if ~isa( varargin{ iOseDensity }, 'nlsaKernelDensity' ) ...
                    || any( size( varargin{ iOseDensity } ) ~= size( varargin{ iOseDenPDistance } ) )
                       error( [ msgId 'invalidDensity' ], ...
                              'The kernelDensity  property must be specified as a nlsaKernelDensity object of size equal to the density pairwise distance' )
                end
                obj.oseDensity = varargin{ iOseDensity };
            else
                obj.oseDensity = nlsaKernelDensity_fb( ...
                                 'partition', partition );
            end
            
            % Delay-embedded kernel density
            if ~isempty( iOseEmbDensity )
                if ~isa( varargin{ iOseEmbDensity }, 'nlsaEmbeddedComponent' ) ...
                  || size( varargin{ iOseEmbDensity }, 1 ) ~= ...
                          numel( obj.oseDenPDistance ) 
                    error( 'The embedded density must be an array of nlsaEmbeddedComponent objects with number of row equal to the number of elements of the denPDistance array' )
                end
                if ~isCompatible( varargin{ iOseEmbDensity }, ...
                                  getOutEmbComponent( obj ), ...
                                  'testComponents', false )
                    error( 'Incompatible embedded density array' )
                end
                if any( getDimension( varargin{ iOseEmbDensity } ) ~= 1 )
                    error( 'Invalid embedded density dimension' )
                end
                obj.oseEmbDensity = varargin{ iOseEmbDensity };
            else
                obj.oseEmbDensity = nlsaEmbeddedComponent_e();
            end

            % OSE pairwise distance
            if ~isempty( iOsePDistance )
                if ~isa( varargin{ iOsePDistance }, 'nlsaPairwiseDistance' ) ...
                       && isscalar( varargin{ iOsePDistance } )
                       error( [ msgId 'invalidOSEPDist' ], ...
                              'The OSE pairwise distance property must be specified as a scalar nlsaPairwiseDistance object' )
                end
                if getNNeighbors( varargin{ iOsePDistance } ) ...
                 > getNTotalSampleTest( varargin{ iOsePDistance } )
                    error( 'The number of OSE nearest neighbors cannot exceed the number of in-sample samples' )
                end
                obj.osePDistance = varargin{ iOsePDistance };
            else
                obj.osePDistance = nlsaPairwiseDistance( ...
                                    'partition', partition, ...
                                    'partitionT', partitionT, ...
                                    'nearestNeighbors', round( nSETot / 10 ) ); 
            end
            nNO = getNNeighbors( obj.osePDistance );

            
            % OSE diffusion operator
            if ~isempty( iOseDiffOp )
                if ~isa( varargin{ iOseDiffOp }, 'nlsaDiffusionOperator_ose' ) ...
                       && isscalar( varargin{ iOseDiffOp } )
                       error( [ msgId 'invalidOSEDIffOp' ], ...
                              'The  property must be specified as a scalar nlsaDiffusionOperator_ose object' )
                end
                if getNNeighbors( varargin{ iOseDiffOp } ) > nNO ...
                  || getNNeighborsTest( varargin{ iOseDiffOp } ) > nNT 
                    error( 'The number of OSE operator nearest neighbors cannot exceed the number of nearest neighbors in the pairwise distances' )
                end 
                obj.oseDiffOp = varargin{ iOseDiffOp };
            else
                obj.oseDiffOp = nlsaDiffusionOperator_ose( ...
                                 'partition', partition, ...
                                 'partitionT', partitionT, ...
                                 'nNeighbors', nNO, ...
                                 'nNeighborsT', nNT, ...
                                 'nEigenfunction', min( 10, nSETot ) ); 
            end

            % OSE target data
            if ~isempty( iOseEmbComponent )
                if ~isa( varargin{ iOseEmbComponent }, 'nlsaEmbeddedComponent_ose' )
                    error( [ msgId 'invalidSclEmb' ], ...
                            'Velocity error  data must be specified as an array of nlsaEmbeddedComponent_ose objects.' )
                end
                
                % Check constistency of data space dimension, embedding indices, and 
                % number of samples
                [ ifC, Test1, Test2 ] = isCompatible( varargin{ iOseEmbComponent }, ...
                                                      obj.outEmbComponent, ...
                                                      'testComponents', false, ...
                                                      'testSamples', true ); 
                if ~ifC
                    msgStr = 'Incompatible OSE component array';
                    disp( Test1 )
                    disp( Test2 )
                    error( [ msgId 'incompatibleOseEmb' ], msgStr )
                end
                obj.oseEmbComponent = varargin{ iOseEmbComponent };
            else
                obj.oseEmbComponent = nlsaEmbeddedComponent_ose( ...
                                        varargin{ iOutTrgEmbComponent } );
            end
         
            % Reconstructed component
            if ~isempty( iOseRecComponent )
                if ~isa( varargin{ iOseRecComponent }, 'nlsaComponent_rec' )
                    error( [ msgId 'invalidOseRecOmponent' ], ...
                        'OSE reconstructed component must be specified as an array of of nlsaComponent_rec objects.' )
                end
                if ~isCompatible( obj.oseEmbComponent, ...
                                  varargin{ iOseRecComponent }, ...
                                  'testSamples', false )
                    error( 'Incompatible OSE reconstructed components' )
                end
                obj.oseRecComponent = varargin{ iOseRecComponent };
            end
        end
    end
         
    methods( Static )
   
        %% LISTCONSTRUCTORPROPERTIES  List property names for class constructor 
        function pNames = listConstructorProperties
            pNames = nlsaModel.listConstructorProperties;
            pNames = [ pNames ...
                       { 'outComponent' ...
                         'outDenComponent' ...
                         'outTrgComponent' ...
                         'outTime' ...
                         'outTimeFormat' ...
                         'outEmbComponent' ...
                         'outDenEmbComponent' ...
                         'outTrgEmbComponent' ...
                         'oseEmbComponent' ...
                         'osePairwiseDistance' ...
                         'oseDenPairwiseDistance' ...
                         'oseKernelDensity' ...
                         'oseEmbKernelDensity' ...
                         'oseDiffusionOperator' ...
                         'oseEmbComponent' ...
                         'oseRecComponent' } ];
        end

        %% LISTPARSERPROPERTIES List property names for class constructor parser
        % NEEDS UPDATING
        function pNames = listParserProperties
            pNames = nlsaModel.listParserProperties;
            pNames = [ pNames ...
                       { 'outComponentName' ...
                         'outRealizationName' ...
                         'outDensityComponentName' ...
                         'outDensityRealizationName' ...
                         'outEmbeddingTemplate' ... 
                         'outEmbeddingPartition' ...
                         'outEmbeddingOrigin' ...
                         'outDenEmbeddingTemplate' ...
                         'outDenEmbeddingOrigin' ...
                         'outDenEmbeddingPartition' ...
                         'oseDenPairwiseDistancePartition' ...
                         'oseDenPairwiseDistanceTemplate' ...
                         'oseKernelDensityTemplate' ...
                         'oseDensityEmbeddingTemplate' ...
                         'osePairwiseDistanceTemplate' ...
                         'oseDiffusionOperatorTemplate' ...
                         'oseEmbeddingTemplate' ...
                         'oseReconstructionPartition' } ];
        end
            
        %% PARSETEMPLATES  Template parser
        propNameVal = parseTemplates( varargin );
    end    
end
