classdef nlsaDiffusionOperator_gl < nlsaDiffusionOperator
%NLSADIFFUSIONOPERATOR Class definition and constructor of diffusion 
%  operator with global data storage
% 
% Modified 2015/05/07    

    %% PROPERTIES
    properties
        fileP     = 'dataP.mat';
        filePhi   = 'dataPhi.mat';
    end

    methods

        %% CLASS CONSTRUCTOR
        function obj = nlsaDiffusionOperator_gl( varargin )

            ifParentArg = true( 1, nargin );
 
            % Parse input arguments
            iFileP         = [];
            iFilePhi       = [];

            for i = 1 : 2 : nargin
                switch varargin{ i }
                    case 'operatorFile'
                        iFileP = i + 1;
                        ifParentArg( [ i i + 1 ] ) = false;
                    case 'eigenfunctionFile'
                        iFilePhi = i + 1;
                        ifParentArg( [ i i + 1 ] ) = false;
                end
            end

            obj = obj@nlsaDiffusionOperator( varargin{ ifParentArg } );

            % Set caller-defined values
            if ~isempty( iFileP )
                if ~isrowstr( varargin{ iFileP } )
                    error( 'Invalid operator file specification' )
                end
                obj.fileP = varargin{ iFileP };
            end
            if ~isempty( iFilePhi )
                if ~isrowstr( varargin{ iFilePhi } )
                    error( 'Invalid eigenfunction file specification' )
                end
                obj.filePhi = varargin{ iFilePhi };
            end
        end
    end
end    
