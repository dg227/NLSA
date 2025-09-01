classdef nlsaSymmetricDistance_gl < nlsaSymmetricDistance
%NLSASYMMETRICDISTANCE_GL Class definition and constructor of NLSA symmetric 
% distances with global storage format
%
% Modified 2014/04/11   

    %% PROPERTIES
    properties
        file  = 'dataYS.mat';
    end

    methods

        %% NLSASYMMETRICDISTANCE_GL Class constructor
        function obj = nlsaSymmetricDistance_gl( varargin )

            ifParentArg = true( 1, nargin );

            % Parse input arguments
            iFile          = [];
            for i = 1 : 2 : nargin
                switch varargin{ i }
                    case 'file'
                        iFile = i + 1;
                        ifParentArg( [ i i + 1 ] ) = false;
                end
            end

            obj = obj@nlsaSymmetricDistance( varargin{ ifParentArg } );

            % Set caller defined values
            if ~isempty( iFile )
                if ischar( varargin{ iFile } )  
                    obj.file = varargin{ iFile };
                else
                    error( 'Invalid data file specification' )
                end
            end
        end
    end
end    
