classdef nlsaKernelOperator
%NLSAKERNELOPERATOR Class definition and constructor of generic kernel 
% operators
%  
% Modified 2020/05/20

    %% PROPERTIES
    properties
        nEig       = 1;
        partition  = nlsaPartition();     % query data
        partitionT = nlsaPartition.empty; % test data
        path       = pwd;
        tag        = '';
    end

    methods

        %% CLASS CONSTRUCTOR
        function obj = nlsaKernelOperator( varargin )

 
            % Parse input arguments
            iNEig          = [];
            iPartition     = [];
            iPartitionT    = [];
            iPath          = [];
            iTag           = [];

            for i = 1 : 2 : nargin
                switch varargin{ i }
                    case 'nEigenfunction'
                        iNEig = 1 + 1;
                    case 'partition'
                        iPartition = i + 1;
                    case 'partitionTest'
                        iPartitionT = i + 1;
                    case 'path'
                        iPath = i + 1;
                    case 'tag'
                        iTag = i + 1;
                    otherwise
                        error( [ 'Invalid property ' varargin{ i } ] )
                end
            end

            % Set caller defined values
            if ~isempty( iPartition )
                if ~isa( varargin{ iPartition }, 'nlsaPartition' ) ...
                        || ~isrow( varargin{ iPartition } )
                    msgStr = [ 'Partition must be specified as a row ' ...
                               'vector of nlsaPartition objects.' ];
                    error( msgStr  )
                end
                obj.partition = varargin{ iPartition };
            end
            if ~isempty( iPartitionT )
                if ~isa( varargin{ iPartitionT }, 'nlsaPartition' ) ...
                        || ~isrow( varargin{ iPartitionT } )
                    msgStr = [ 'Test partition must be specified as a row ' ...
                               'vector of nlsaPartition objects.' ];
                    error( msgStr )
                end
                obj.partitionT = varargin{ iPartitionT };
            end
            if ~isempty( iNEig )
                if ~ispsi( varargin{ iNEig } )
                    msgStr = [ 'Number of eigenfunctions must be a ' ...
                               'positive scalar integer.' ]; 
                    error( msgStr )
                end
                obj.nEig = varargin{ iNEig };
            end
            if ~isempty( iPath )
                if ~isrowstr( varargin{ iPath } )
                    error( 'Invalid path specification' )
                end
                obj.path = varargin{ iPath };
            end
            if ~isempty( iTag )
                if ~iscell( varargin{ iTag } ) ...
                  || ~isrowstr( varargin{ iTag } )
                    error( 'Invalid object tag' )
                end
                obj.tag = varargin{ iTag };
            end
        end
    end

    methods( Abstract )

        %% COMPUTEOPERATOR Compute kernel matrix elements
        computeOperator( obj )

        %% COMPUTEEIGENFUNCTIONS 
        computeEigenfunctions( obj )

        %% GETOPERATOR Get operator matrix elements
        p = getOperator( obj )

        %% GETEIGENFUNCTIONS Get eigenfunctions
        phi = getEigenfunctions( obj )
    end


    methods( Static )

        %% ISVALIDIDX Helper function to validate basis function indices 
        function ifV = isValidIdx( idx )
            ifV = true;

            if ~isvector( idx )
                ifV = false;
                return
            end

            idx  = sort( idx, 'ascend' );
            if ~ispsi( idx( 1 ) )
                ifV = false;
                return
            end
            if ~ispi( idx( 2 : end ) - idx( 1 : end - 1 ) )
                ifV = false;
            end
        end
    end
end    
