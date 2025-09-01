classdef nlsaKernelDensity_vb < nlsaKernelDensity_fb
%NLSAKERNELDENSITY Class definition and constructor of variable-bandwidth 
%  kernel density estimation
% 
% Modified 2015/04/07

    %% PROPERTIES
    properties
        fileRho = 'dataRho.mat';
        kNN = 8;
    end

    methods

        %% CLASS CONSTRUCTOR
        function obj = nlsaKernelDensity_vb( varargin )

            ifParentArg = true( 1, nargin );
 
            % Parse input arguments
            iFileRho = [];
            iKNN     = [];

            for i = 1 : 2 : nargin
                switch varargin{ i }
                    case 'kernelNormalizationFile'
                        iFileRho = i + 1;
                        ifParentArg( [ i i + 1 ] ) = false;
                    case 'kNN'
                        iKNN = i + 1;
                        ifParentArg( [ i i + 1 ] ) = false;
                end
            end

            obj = obj@nlsaKernelDensity_fb( varargin{ ifParentArg } );

            % Set caller-defined values
            if ~isempty( iFileRho )
                if ~isrowstr( iFileRho )
                    error( 'Invalid normalization file specification' )
                end
                obj.fileRho = varargin{ iFileRho };
            end
            if ~isempty( iKNN )
                if ~ispsi( varargin{ iKNN } )
                    error( 'Number of nearest neighbors must be a positive scalar' )
                end
                obj.kNN = varargin{ iKNN };
            end
        end
    end
end    
