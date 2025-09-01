classdef nlsaModel_err < nlsaModel_ose
%NLSAMODEL_ERR   Class definition and constructor of NLSA model supporting
% error between in-sample (IS, "nature") and out-of-sample (OS, "model") data
%
%
% outPDistance:       Pairwise distance for OS data
% outSDistance:       Symmetric distance for OS data
% outDiffOp:          Diffusion operator on model data
% outPrjComponent:    Projected target OS data onto eigenfunctions of outDiffOp 
% oseRefComponent:    Reference data on the OS manifold
% oseEmbComponent:    trgEmbComponent extended to the OS manifold
% oseErrComponent:    Difference between oseEmbComponent and oseRefComponent

% isrRefComponent:       Reference data on the in-sample (IS) manifold. 
% isrPDistance:       Pairwise distance, IS relative to OS
% isrDiffOp:          Restriction operator from OS to IS manifold
% isrEmbComponent:    outTrgEmbComponent mapped to IS manifold
% isrErrComponent:    Difference between isrEmbComponent and isrRefComponent 

% Modified 2014/10/13

    %% PROPERTIES
    properties
        outTrgTime         = { 1 };
        outTrgComponent    = nlsaComponent();
        outTrgEmbComponent = nlsaEmbeddedComponent_e();
        outPDistance       = nlsaPairwiseDistance();    
        outSDistance       = nlsaSymmetricDistance_gl();
        outDiffOp          = nlsaDiffusionOperator_gl();
        outPrjComponent    = nlsaProjectedComponent();
        oseRefComponent    = nlsaEmbeddedComponent_e();
        oseErrComponent    = nlsaEmbeddedComponent_d();
        isrPDistance       = nlsaPairwiseDistance();
        isrDiffOp          = nlsaDiffusionOperator_ose(); 
        isrEmbComponent    = nlsaEmbeddedComponent_ose(); % nature rel. 
        isrErrComponent    = nlsaEmbeddedComponent_d
        isrRefComponent    = nlsaEmbeddedComponent_e();
    end
    
    methods
        
        %% CLASS CONSTRUCTOR
        function obj = nlsaModel_err( varargin )
            
            msgId = 'nlsa:nlsaModel_err:';
            
            % Check if constructor is called in "template" mode and parse
            % templates if needed
            if ifTemplate( 'nlsaModel_err', varargin{ : } )
                varargin = nlsaModel_err.parseTemplates( varargin{ : } );
                nargin   = numel( varargin );
            end
  
            nargin = numel( varargin );
            ifParentArg = true( 1, nargin ); 

            % Parse input arguments 
            iOutTrgTime         = [];
            iOutTrgComponent    = [];
            iOutTrgEmbComponent = [];
            iOutPDistance       = [];
            iOutSDistance       = [];
            iOutDiffOp          = [];
            iOutPrjComponent    = [];
            iOseRefComponent    = [];
            iOseErrComponent    = [];
            iIsrPDistance       = [];
            iIsrDiffOp          = [];
            iIsrEmbComponent    = [];
            iIsrRefComponent    = [];
            iIsrErrComponent    = [];

            for i = 1 : 2 : nargin
                switch varargin{ i }
                    case 'outTrgTime'
                        iOutTrgTime = i + 1;
                        ifParentArg( [ i i + 1 ] ) = false;
                    case 'outTrgComponent'
                        iOutTrgComponent = i + 1;
                        ifParentArg( [ i i + 1 ] ) = false;
                    case 'outTrgEmbComponent'
                        iOutTrgEmbComponent = i + 1;
                        ifParentArg( [ i i + 1 ] ) = false;
                    case 'outPairwiseDistance'
                        iOutPDistance = i + 1;
                        ifParentArg( [ i i + 1 ] ) = false;
                    case 'outSymmetricDistance'
                        iOutSDistance = i + 1;
                        ifParentArg( [ i i + 1 ] ) = false;
                    case 'outDiffusionOperator'
                        iOutDiffOp = i + 1;
                        ifParentArg( [ i i + 1 ] ) = false;
                    case 'outPrjComponent'
                        iOutPrjComponent = i + 1;
                        ifParentArg( [ i i + 1 ] ) = false;
                    case 'oseRefComponent'
                        iOseRefComponent = i + 1;
                        ifParentArg( [ i i + 1 ] ) = false;
                    case 'oseErrComponent'
                        iOseErrComponent = i + 1;
                        ifParentArg( [ i i + 1 ] ) = false;
                    case 'isrPairwiseDistance' 
                        iIsrPDistance = i + 1;
                        ifParentArg( [ i i + 1 ] ) = false;
                    case 'isrDiffusionOperator'
                        iIsrDiffOp = i + 1;
                        ifParentArg( [ i i + 1 ] ) = false;
                    case 'isrEmbComponent'
                        iIsrEmbComponent = i + 1;
                        ifParentArg( [ i i + 1 ] ) = false;
                    case 'isrErrComponent'
                        iIsrErrComponent = i + 1;
                        ifParentArg( [ i i + 1 ] ) = false;
                    case 'isrRefComponent'
                        iIsrRefComponent = i + 1;
                        ifParentArg( [ i i + 1 ] ) = false;
                end
            end

            obj = obj@nlsaModel_ose( varargin{ ifParentArg } ); 

            nS  = getNEmbSample( obj );
            nSO = getNOutEmbSample( obj );
            nCT = size( obj.outTrgComponent, 1 );
           

            % OSE target components 
            if ~isempty( iOutTrgComponent )
                if ~isa( varargin{ iOutTrgComponent }, 'nlsaComponent' )
                    error( [ msgId 'invalidTrg' ], ...
                           'Target data must be specified as an array of nlsaComponent objects.' )
                end
                [ ifC, Test1, Test2 ] = isCompatible( varargin{ iOutTrgComponent }, ...
                                                      varargin{ iOutComponent }, ...
                                                      'testComponents', false, ...
                                                      'testSamples', true );
                if ~ifC
                    disp( Test1 )
                    disp( Test2 )
                    error( [ msgId, 'notCompatibleTrg' ], ...
                           'Incompatible target component array' )
                end
                obj.outTrgComponent = varargin{ iOutTrgComponent };
  
           else
                obj.outTrgComponent = obj.outComponent;
            end     
            [ nCT, nRT ] = size( obj.outTrgComponent );
            nSRT = getNSample( obj.outTrgComponent( 1, : ) );   % number of samples in each realization

            % Time for target data
            if ~isempty( iOutTrgTime )
                if isvector( varargin{ iOutTrgTime } ) && ~iscell( varargin{ iOutTrgTime } )
                    obj.outTrgTime = cell( 1, nRT );
                    for iR = 1 : nRT
                        obj.outTrgTime{ iR } = varargin{ iOutTrgTime };
                    end
                elseif   isvector( varargin{ iOutTrgTime } ) && iscell( varargin{ iOutTrgTime } ) ...
                      && numel( varargin{ iOutTrgTime } ) == nRT
                    obj.outTrgTime = varargin{ iOutTrgTime };
                else
                    error( [ msgId 'invalidOutTrgTime' ], ...
                               'Target time data must be either a vector or a cell vector of size nRT.' )
                end
                if size( obj.outTrgTime, 1 ) > 1
                    obj.outTrgTime = obj.outTrgTime';
                end
                for iR = 1 : nRT
                    if ~isvector( obj.outTrgTime{ iR } ) 
                        msgStr = sprintf( [ 'Invalid timestamp array for realization %i: \n' ... 
                                            'Expecting vector \n' ...
                                            'Received  array of size [%i]' ], ...  
                                            iR, nsR( iR ), size( obj.outTrgTime{ iR } ) );  
                        error( [ msg 'invalidTimestamps' ], msgStr )
                    end
                    if numel( obj.outTrgTime{ iR } ) ~= nSR( iR )
                        msgStr = sprintf( [ 'Invalid number of timestamps for realization %i: \n' ... 
                                            'Expecting %i \n' ...
                                            'Received  %i \n' ], ...  
                                            iR, nsR( iR ), numel( obj.outTrgTime{ iR } ) );  
                        error( [ msg 'invalidOutTrgTimestamps' ], msgStr )
                    end
                    if ~ismonotonic( obj.outTrgTime{ iR }, [], 'increasing' )
                        msgStr = sprintf( [ 'Invalid timestamps for realization %i: \n' ... 
                                            'Time values must be monotonically increasing.' ], ...
                                          iR );
                        error( [ msg 'invalidOutTrgTimestamps' ], msgStr )
                    end
                    if size( obj.outTrgTime{ iR }, 1 ) > 1
                        obj.outTrgTime{ iR } = obj.outTrgTime{ iR }';
                    end
                end
            else % set to source timestamps if no caller input 
                obj.outTrgTime = obj.outTime;
            end

            % Target embedded components
            % Check input array class and size
            if ~isempty( iOutTrgEmbComponent )
                if ~isa( varargin{ iOutTrgEmbComponent }, 'nlsaEmbeddedComponent' )
                    error( [ msgId 'invalidTrgEmb' ], ...
                           'Target embedded data must be specified as an array of nlsaEmbeddedComponent objects.' )
                end
                obj.outTrgEmbComponent = varargin{ iOutTrgEmbComponent };
            else
                obj.outTrgEmbComponent = obj.outEmbComponent;
            end
            % Check constistency of physical space dimension, embedding indices dimension, and 
            % number of samples
            [ ifC, Test1, Test2 ] = isCompatible( obj.oseEmbComponent, ...
                                                  varargin{ iOutTrgEmbComponent }, ...
                                                  'testComponents', true, ...
                                                  'testSamples', true );
            if ~ifC
                disp( Test1 )
                disp( Test2 )
                error( [ msgId 'incompatibleSrcTrgEmb' ], ...
                       'Incompatible target embedded component arrays' )
            end                

            % OS pairwise distance
            if ~isempty( iOutPDistance )
                if ~isa( varargin{ iOutPDistance }, 'nlsaPairwiseDistance' ) ...
                       && ~isscalar( varargin{ iOutPDistance } )
                       error( [ msgId 'invalidModPDist' ], ...
                              'The model pairwise distance property must be specified as a scalar nlsaPairwiseDistance object' )
                end
                obj.outPDistance = varargin{ iOutPDistance };
            else
                obj.outPDistance = nlsaPairwiseDistance( ...
                                    'partition', getPartition( getOSEPDistance( obj ) ), ...
                                    'nearestNeighbors', round( nSO / 10 ) ); 
            end

            % OS symmetric distance
            if ~isempty( iOutSDistance )
                if ~isa( varargin{ iOutSDistance }, 'nlsaSymmetricDistance' ) ...
                       && isscalar( varargin{ iOutSDistance } )
                       error( [ msgId 'invalidSDist' ], ...
                              'The model symmetric distance property must be specified as a scalar nlsaSymmetricDistance object' )
                end
                obj.outSDistance = varargin{ iOutSDistance };
            else
                obj.outSDistance = nlsaSymmetricDistance_gl( ...
                                     'partition', getPartition( obj.outPDistance ), ...
                                     'nearestNeighbors', getNNeighbors( obj.outPDistance ) );
            end
            
            % OS diffusion operator
            if ~isempty( iOutDiffOp )
                if ~isa( varargin{ iOutDiffOp }, 'nlsaDiffusionOperator' ) ...
                       && isscalar( varargin{ iOutDiffOp } )
                       error( [ msgId 'invalidOutDiffOp' ], ...
                              'The model diffusion operator property must be specified as a scalar nlsaDiffusionOperator object' )
                end
                obj.outDiffOp = varargin{ iOutDiffOp };
            else
                obj.outDiffOp = nlsaDiffusionOperator_gl( ...
                                  'partition', getPartition( obj.modPDist ), ...
                                  'nEigenfunction', min( 10, nSO ) ); 
            end
            

            % OS projected component
            if ~isempty( iOutPrjComponent )
                if ~isa( varargin{ iOutPrjComponent }, 'nlsaProjectedComponent' )
                    error( [ msgId 'invalidPrj' ], ...
                        'Projected data must be specified as nlsaProjectedComponent objects' )
                end
                if ~isCompatible( varargin{ iOutPrjComponent }, obj.outTrgEmbComponent )
                    error( 'Incompatible projected components' )
                end
                obj.outPrjComponent = varargin{ iOutPrjComponent };
            else
                nL  = getNEigenfunction( obj.outDiffOp );
                for iC = nCT : -1 : 1
                    nDE = getEmbeddingSpaceDimension( obj.outTrgEmbComponent( iC ) );
                    obj.outPrjComponent( iC ) = nlsaProjectedComponent( ...
                        'embeddingSpaceDimension', nDE, ...
                        'partition', getPartition( obj.modPDist ), ...
                        'nBasisFunction', nL );
                end
                obj.outPrjComponent = obj.outPrjComponent';
            end

            % Reference components for OSE error calculation  
            if ~isempty( iOseRefComponent )
                if ~isa( varargin{ iOseRefComponent }, 'nlsaEmbeddedComponent' )
                    error( [ msgId 'invalidTrgEmb' ], ...
                           'Target embedded data must be specified as an array of nlsaEmbeddedComponent objects.' )
                end
                % Check constistency of physical space dimension, embedding indices dimension, and 
                % number of samples
                [ ifC, Test1, Test2 ] = isCompatible( ...
                    obj.outTrgEmbComponent, ...
                    varargin{ iOseRefComponent }, ...
                    'testComponents', true, ...
                    'testSamples', true );

                if ~ifC
                    disp( Test1 )
                    disp( Test2 )
                    error( [ msgId 'incompatibleOutRef' ], ...
                           'Incompatible reference component arrays' )
                end                

                obj.oseRefComponent = varargin{ iOseRefComponent };


            else
                obj.oseRefComponent = obj.outTrgEmbComponent;
            end
            
            % OSE error (IS data relative to OS data)
            if ~isempty( iOseErrComponent )
                if ~isa( varargin{ iOseErrComponent }, 'nlsaEmbeddedComponent_d' )
                    error( [ msgId 'invalidOseErr' ], ...
                            'Velocity error  data must be specified as an array of nlsaEmbeddedComponent_d objects.' )
                end
                
                % Check constistency of data space dimension, embedding indices, and 
                % Number of samples
                [ ifC, Test1, Test2 ] = isCompatible( varargin{ iOseErrComponent }, ...
                                                      obj.oseRefComponent, ...
                                                      'testComponents', true, ...
                                                      'testSamples', true ); 
                if ~ifC
                    msgStr = 'Incompatible OSE error component array';
                    disp( Test1 )
                    disp( Test2 )
                    error( [ msgId 'incompatibleSclEmb' ], msgStr )
                end
                obj.oseErrComponent = varargin{ iOseErrComponent };
            else
                obj.oseErrComponent = nlsaEmbeddedComponent_d( ...
                                        varargin{ iOutTrgEmbComponent } );
            end
         

            % Pairwise distances for ISR operator (OS -> IS)
            if ~isempty( iIsrPDistance )
                if ~isa( varargin{ iIsrPDistance }, 'nlsaPairwiseDistance' ) ...
                       && isscalar( varargin{ iIsrPDistance } )
                       error( [ msgId 'invalidOSEPDist' ], ...
                              'The OSE pairwise distance property must be specified as a scalar nlsaPairwiseDistance object' )
                end
                obj.isrPDistance = varargin{ iIsrPDistance };
            else
                obj.isrPDistance = nlsaPairwiseDistance( ...
                                    'partition', getPartitionTest( getOSEPairwiseDistance( obj ) ), ...
                                    'partitionT', getPartition( getOSEPairwiseDistance( obj ) ), ...
                                    'nearestNeighbors', round( nSO / 10 ) ); 
            end

            % ISR diffusion operatror
            if ~isempty( iIsrDiffOp )
                if ~isa( varargin{ iIsrDiffOp }, 'nlsaDiffusionOperator_ose' ) ...
                       && isscalar( varargin{ iIsrDiffOp } )
                       error( [ msgId 'invalidOSEDIffOp' ], ...
                              'The  property must be specified as a scalar nlsaDiffusionOperator_ose object' )
                end
                obj.isrDiffOp = varargin{ iIsrDiffOp };
            else
                obj.isrDiffOp = nlsaDiffusionOperator_ose( ...
                                    'partition', getPartition( obj.isrPDistance ), ...
                                    'partitionT', getPartitionTest( obj.isrPDistance ), ...
                                    'nEigenfunction', min( 10, nS ) ); 
            end

            % ISR data to IS manifold
            if ~isempty( iIsrEmbComponent )
                if ~isa( varargin{ iIsrEmbComponent }, 'nlsaEmbeddedComponent_ose' )
                    error( [ msgId 'invalidSclEmb' ], ...
                            'Velocity error  data must be specified as an array of nlsaEmbeddedComponent_ose objects.' )
                end
                
                % Check constistency of data space dimension, embedding indices, and 
                % Number of samples
                [ ifC, Test1, Test2 ] = isCompatible( varargin{ iIsrEmbComponent }, ...
                                                      getTrgEmbComponent( obj ), ...
                                                      'testComponents', true, ...
                                                      'testSamples', true ); 
                if ~ifC
                    msgStr = 'Incompatible ISR component array';
                    disp( Test1 )
                    disp( Test2 )
                    error( [ msgId 'incompatibleIsrEmb' ], msgStr )
                end
                obj.isrEmbComponent = varargin{ iIsrEmbComponent };
            else
                obj.isrEmbComponent = nlsaEmbeddedComponent_ose( ...
                                        getTrgEmbComponent( obj ) );
            end
                

            % Reference components for ISR error  
            if ~isempty( iIsrRefComponent )
                if ~isa( varargin{ iIsrRefComponent }, 'nlsaEmbeddedComponent' )
                    error( [ msgId 'invalidRef' ], ...
                           'ISR reference data must be specified as an array of nlsaEmbeddedComponent objects.' )
                end
                obj.isrRefComponent = varargin{ iIsrRefComponent };
            else
                obj.isrRefComponent = obj.trgEmbComponent;
            end
            % Check constistency of physical space dimension, embedding indices dimension, and 
            % number of samples
            [ ifC, Test1, Test2 ] = isCompatible( obj.trgEmbComponent, ...
                                                  varargin{ iIsrRefComponent }, ...
                                                  'testComponents', true, ...
                                                  'testSamples', true );
            if ~ifC
                disp( Test1 )
                disp( Test2 )
                error( [ msgId 'incompatibleRef' ], ...
                       'Incompatible reference component arrays' )
            end                

            % ISR error (OS data relative to IS data)
            if ~isempty( iIsrErrComponent )
                if ~isa( varargin{ iIsrErrComponent }, 'nlsaEmbeddedComponent_d' )
                    error( [ msgId 'invalidOseErr' ], ...
                            'Velocity error  data must be specified as an array of nlsaEmbeddedComponent_d objects.' )
                end
                
                % Check constistency of data space dimension, embedding indices, and 
                % Number of samples
                [ ifC, Test1, Test2 ] = isCompatible( varargin{ iIsrErrComponent }, ...
                                                      obj.oseRefComponent, ...
                                                      'testComponents', true, ...
                                                      'testSamples', true ); 
                if ~ifC
                    msgStr = 'Incompatible OSE error component array';
                    disp( Test1 )
                    disp( Test2 )
                    error( [ msgId 'incompatibleSclEmb' ], msgStr )
                end
                obj.isrErrComponent = varargin{ iIsrErrComponent };
            else
                obj.isrErrComponent = nlsaEmbeddedComponent_d( ...
                                        obj.trgEmbComponent );
            end
 
          
        end
    end
         
    methods( Static )
   
        %% LISTCONSTRUCTORPROPERTIES  List property names for class constructor  
        function pNames = listConstructorProperties
            pNames = nlsaModel_ose.listConstructorProperties;
            pNames = [ pNames ...
                      {  'outTrgTime' ...
                         'outTrgComponent' ...
                         'outTrgEmbComponent' ...
                         'outPairwiseDistance' ...
                         'outSymmetricDistance' ...
                         'outDiffusionOperator' ...
                         'outPrjComponent' ...
                         'oseRefComponent' ...
                         'oseErrComponent' ...
                         'isrPairwiseDistance' ...
                         'isrDiffusionOperator' ...
                         'isrEmbComponent' ...
                         'isrRefComponent' ...
                         'isrErrComponent' } ];
        end

        %% LISTPARSERPROPERTIES List property names for class constructor parser
        function pNames = listParserProperties
            pNames = nlsaModel_ose.listParserProperties;
            pNames = [ pNames ...
                       { 'sclPairwiseDistanceTemplate' ...
                         'sclSymmetricDistanceTemplate' ...
                         'sclDiffusionOperatorTemplate' ...
                         'modPairwiseDistanceTemplate' ...
                         'modSymmetricDistanceTemplate' ...
                         'modDiffusionOperatorTemplate' ...
                         'modPairwiseDistanceTemplate' ...
                         'modSymmetricDistanceTemplate' ...
                         'prjPairwiseDistanceTemplate' ...
                         'prjDiffusionOperatorTemplate' } ];
        end
            
        %% PARSETEMPLATES  Template parser
        propNameVal = parseTemplates( varargin );
    end    
end
