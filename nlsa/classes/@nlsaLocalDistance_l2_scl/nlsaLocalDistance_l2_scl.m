classdef nlsaLocalDistance_l2_scl < nlsaLocalDistance_l2
%NLSADISTANCE_L2_SCL  Class definition and constructor of L2 
% distance function with scaling
%
% Modified 2015/10/30    

    %% PROPERTIES
    properties
        lScaling = nlsaLocalScaling_exp();
    end

    methods
        %% NLSADISTANCE_L2_SCL  Class constructor
        function obj = nlsaLocalDistance_l2_scl( varargin )
            
            ifParentArg = true( 1, nargin );

            % Parse input arguments
            iLScaling = [];
            for i = 1 : 2 : nargin
                switch varargin{ i }
                    case 'localScaling'
                        iLScaling = i + 1;
                        ifParentArg( [ i i + 1 ] ) = false;
                end
            end

            obj = obj@nlsaLocalDistance_l2( varargin{ ifParentArg } );
           
            % Set caller-defined values
            if ~isempty( iLScaling )
                if ~isa( varargin{ iLScaling }, 'nlsaLocalScaling' )
                    error( 'Scaling property must be set to an nlsaLocalScaling object' )
                end
                obj.lScaling = varargin{ iLScaling };
            end
        end
    end
end
