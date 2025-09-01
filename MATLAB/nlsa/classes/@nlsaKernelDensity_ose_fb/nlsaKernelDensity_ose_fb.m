classdef nlsaKernelDensity_ose_fb < nlsaKernelDensity_ose
%NLSAKERNELDENSITY_OSE_FB Class definition and constructor of fixed-bandwidth 
%  kernel density estimation and out-of-sample data
% 
% Modified 2018/07/05

    %% PROPERTIES
    properties
        fileQ       = 'dataQ.mat';
        pathQ       = 'pathQ';
    end

    methods

        %% CLASS CONSTRUCTOR
        function obj = nlsaKernelDensity_ose_fb( varargin )

            ifParentArg = true( 1, nargin );
 
            % Parse input arguments
            iFileQ       = [];
            iPathQ       = [];

            for i = 1 : 2 : nargin
                switch varargin{ i }
                    case 'densityFile'
                        iFileQ = i + 1;
                        ifParentArg( [ i i + 1 ] ) = false;
                    case 'densitySubpath'
                        iPathQ = i + 1;
                        ifParentArg( [ i i + 1 ] ) = false;
                end
            end

            obj = obj@nlsaKernelDensity_ose( varargin{ ifParentArg } );

            if ~isempty( iFileQ )
                if ~isrowstr( varargin{ iFileQ } )
                    error( 'Invalid density file specification' )
                end
                obj.fileQ = varargin{ iFileQ };
            end
            if ~isempty( iPathQ )
                if ~isrowstr( varargin{ iPathQ } )
                    error( 'Invalid density subpath specification' )
                end
                obj.pathQ = varargin{ iPathQ };
            end
        end
    end
end    
