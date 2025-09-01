classdef nlsaCovarianceOperator < nlsaKernelOperator
%NLSACOVARIANCEOPERATOR Class definition and constructor of 
% covariance operator 
%
% Modified 2014/07/16    

    properties
        partitionD = nlsaPartition(); % nlsaPartitionObject specifying the spatial indices of the components 
        fileU    = nlsaFilelist(); % left (spatial) singular vector file
        fileV    = nlsaFilelist(); % right (temporal) singular vector file
        fileS    = 'dataS.mat';    % singular values file
        fileEnt  = 'dataEnt.mat';  % spectral entropy file
        pathA    = 'dataA';        % operator subpath 
        pathU    = 'dataU';        % left singular vector subpath 
        pathV    = 'dataV';        % right singular vector subpath
        pathS    = 'dataS';        % singular value file
    end

    methods

        function obj = nlsaCovarianceOperator( varargin )

            ifParentArg = true( 1, nArgin );

            % Parse input arguments
            iFileU         = [];
            iFileS         = [];
            iFileV         = [];
            iFileEnt       = [];
            iPartitionD    = [];
            iPathA         = [];
            iPathU         = [];
            iPathV         = [];
            iPathS         = [];
            for i = 1 : 2 : nargin
                switch varargin{ i }
                    case 'spatialPartition'
                        iPartitionD = i + 1;
                        ifParentArg( [ i i + 1 ] ) = true;
                    case 'leftSingularVectorFIle'
                        iFileU = i + 1;
                        ifParentArg( [ i i + 1 ] ) = true;
                    case 'singularValueFile'
                        iFileS = i + 1;
                        ifParentArg( [ i i + 1 ] ) = true;
                    case 'rightSingularVectorFile'
                        iFileV = i + 1;
                        ifParentArg( [ i i + 1 ] ) = true;
                    case 'spectralEntropyFile'
                        iFileEnt = i + 1;	
                        ifParentArg( [ i i + 1 ] ) = true;
                    case 'operatorSubpath'
                        iPathA = i + 1;
                        ifParentArg( [ i i + 1 ] ) = true;
                end
            end

            obj = obj@nlsaKernelOperator( varargin{ ifParentArg } );

            % Set caller defined values
            if ~isempty( iPartitionD )
                if ~isa( varargin{ iPartitionD }, 'nlsaPartition' ) ...
                  || ~iscolumn( varargin{ iPartitionD } )
                    error( 'Spatial partition must be specified as a column vector of nlsaPartition objects' )
                end
                obj.partitionD = varargin{ iPartition };
            end 
            if ~isempty( iFileV )
                if ~isa( varargin{ iFileV }, 'nlsaFilelist' ) ...
                  || ~isCompatible( varargin{ iFileV }, obj.partition )
                    error( 'Right singular vector file property must be set to a vector of nlsaFilelist objects compatible with the query partition' )
                end
                obj.fileV = varargin{ iFileV };
            else
                obj.fileV = nlsaFilelist( obj.partition );
            end
            if ~isempty( iFileU )
                if ~isa( varargin{ iFileU }, 'nlsaFilelist' ) ...
                  || getNFile( varargin{ iFileU } ~= getNBatch( obj.partitionD ) 
                    error( 'Left singular vector file property must be set to an nlsaFilelist object with number of files equal to the number of components' )
                end
                obj.fileU = varargin{ iFileU };
            else
                obj.fileU = nlsaFilelist( obj.partitionD );
            end
            if ~isempty( iFileS )
                if ~isrowstr( varargin{ iFileS } ) 
                    error( 'Invalid singular value file specification' )
                end
                obj.fileS = varargin{ iFileS };
            end
            if ~isrowstr( iFileEnt )
                if ~isrowstr( varargin{ iFileEnt } ) 
                    error( 'Invalid spectral entropy file specification' )
                end
                obj.fileEnt = varargin{ iFileEnt };
            end
            if ~isempty( iPathA )
                if ~isrowstr( varargin{ iPathA } )
                    error( 'Invalid operator subpath path specification' )
                end
                obj.pathA = varargin{ iPathA }; 
            end
            if ~isempty( iPathU )
                if ~isrowstr( varargin{ iPathU } )
                    error( 'Invalid left singular vector subpath specification' )
                end
                obj.pathU = varargin{ iPathU }; 
            end
            if ~isempty( iPathV )
                if ~isrowstr( varargin{ iPathV } )
                    error( 'Invalid right singular vector subpath specification' )
                end
                obj.pathV = varargin{ iPathV }; 
            end
            if ~isempty( iPathS )
                if ~isrowstr( varargin{ iPathS } )
                    error( 'Invalid singular value subpath specification' )
                end
                obj.pathS = varargin{ iPathS }; 
            end
        end
    end
end    
