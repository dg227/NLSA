classdef nlsaSymmetricDistance_batch < nlsaSymmetricDistance
%NLSASYMMETRICDISTANCE_BATCH Class definition and constructor of NLSA symmetric 
% distances with batch storage format
%
% Modified 2014/05/01   

    %% PROPERTIES
    properties
        file  = nlsaFilelist();
        nNMax = 3;
    end

    methods

        %% NLSASYMMETRICDISTANCE_BATCH Class constructor
        function obj = nlsaSymmetricDistance_batch( varargin )

            ifParentArg = true( 1, nargin );

            % Parse input arguments
            iFile  = [];
            iNNMax = [];
            for i = 1 : 2 : nargin
                switch varargin{ i }
                    case 'file'
                        iFile = i + 1;
                        ifParentArg( [ i i + 1 ] ) = false;
                    case 'nNeighborsMax'
                        iNNMax = i + 1;
                        ifParentArg( [ i i + 1 ] ) = false;
                end
            end

            obj = obj@nlsaSymmetricDistance( varargin{ ifParentArg } );

            % Set caller defined values
            if ~isempty( iFile )
                if ~isa( varargin{ iFile }, 'nlsaFilelist' ) ...
                   || ~iscompatible( varargin{ iFile }, obj.partition )                        
                     error( 'File property must be set to a vector of nlsaFilelist objects compatible with the query partition' )
                end
                obj.file = varargin{ iFile };
            else
                obj.file = nlsaFilelist( obj.partition );
            end
            if ~isempty( iNNMax )
                if ~ispsi( varargin{ iNNMax } ) ...
                  || varargin{ iNNMax } < getNNeighbors( obj )
                    error( 'The maximum number of nearest neighbors must be a positive scalar integer greater or equal than the number of nearest neighbors' )
                end
                obj.nNMax = varargin{ iNNMax };
            else
                obj.nNMax = round( 1.5 * getNNeighbors( obj ) );
            end
        end
    end
end    
