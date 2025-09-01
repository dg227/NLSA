classdef nlsaKernelDensity_ose_vb < nlsaKernelDensity_ose_fb
%NLSAKERNELDENSITY_OSE_VB Class definition and constructor of variable-bandwidth 
%  kernel density estimation for out-of-sample data
% 
% Modified 2018/07/06

    %% PROPERTIES
    properties
        fileRho = 'dataRho.mat';
        kNN = 8;
    end

    methods

        %% CLASS CONSTRUCTOR
        function obj = nlsaKernelDensity_ose_vb( varargin )

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

            obj = obj@nlsaKernelDensity_ose_fb( varargin{ ifParentArg } );

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
