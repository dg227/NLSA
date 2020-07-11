classdef nlsaLocalDistance_l2 < nlsaLocalDistance
%NLSALOCALDISTANCE_L2  Class definition and constructor of local distance 
% based on L2 norm
%
% nPar is the number of workers used in parallel for loops to compute 
% lagged distances

% Modified 2020/07/11   

    %% PROPERTIES
    properties
        nPar = 0; 
    end

    methods

        %% NLSALOCALDISTANCE_L2  Class constructor
        function obj = nlsaLocalDistance_l2( varargin )

            ifParentArg = true( 1, nargin );

            % Parse input arguments
            iNPar  = [];

            for i = 1 : 2 : nargin
                switch varargin{ i }
                case 'nPar'
                    iNPar = i + 1;
                    ifParentArg( [ i i + 1 ] ) = false;
                end
            end

            obj = obj@nlsaLocalDistance( varargin{ ifParentArg } );

            % Set caller-defined values
            if ~isempty( iNPar )
                if ~isnnsi( varargin{ iNPar } )
                    msgStr = [ 'Number of parallel workers must be a ' ...
                               'nonnegative scalar integer.' ]; 
                    error( msgStr )
                end
                obj.nPar = varargin{ iNPar };
            end
        end
                        
    end
end    
