classdef nlsaKernelDensity
%NLSAKERNELDENSITY Class definition and constructor of generic kernel 
% densities
%  
% Modified 2018/07/05

    %% PROPERTIES
    properties
        nD         = 1;
        epsilon    = 1;
        partition  = nlsaPartition();   % query and out-of-sample data
        path       = pwd;
        tag        = '';
    end

    methods

        %% CLASS CONSTRUCTOR
        function obj = nlsaKernelDensity( varargin )

 
            % Parse input arguments
            iD             = [];
            iEpsilon       = [];
            iPartition     = [];
            iPath          = [];
            iTag           = [];

            for i = 1 : 2 : nargin
                switch varargin{ i }
                    case 'epsilon'
                        iEpsilon = i + 1;
                    case 'dimension'
                        iD = i + 1;
                    case 'partition'
                        iPartition = i + 1;
                    case 'path'
                        iPath = i + 1;
                    case 'tag'
                        iTag = i + 1;
                    otherwise
                        error( [ 'Invalid property ' varargin{ i } ] )
                end
            end

            % Set caller-defined values
            if ~isempty( iEpsilon )
                if ~isps( varargin{ iEpsilon } )
                    error( 'The bandwidth parameter must be a positive scalar' )
                end
                obj.epsilon = varargin{ iEpsilon };
            end
            if ~isempty( iD )
                if ~isps( varargin{ iD } )
                    error( 'Invalid component dimension' )
                end
                obj.nD = varargin{ iD };
            end
            if ~isempty( iPartition )
                if ~isa( varargin{ iPartition }, 'nlsaPartition' ) ...
                        || ~isrow( varargin{ iPartition } )
                    error( 'Partition must be specified as a row vector of nlsaPartition objects' )
                end
                obj.partition = varargin{ iPartition };
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

        %% COMPUTEDENSITY
        computeDensity( obj )

        %% GETDENSITY 
        q = getDensity( obj )
    end
end    
