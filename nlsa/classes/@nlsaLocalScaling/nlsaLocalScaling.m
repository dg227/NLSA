classdef nlsaLocalScaling
%NLSALOCALSCALING  Class definition and constructor of local scaling factors
% used in kernels
%
% Modified 2015/10/31  

   %% PROPERTIES
    properties
        mode  = 'explicit';
        tag   = '';
    end

    methods

        %% NLSALOCALSCALING  Class contructor
        function obj = nlsaLocalScaling( varargin )

            % Parse input arguments
            iTag  = [];
            iMode = [];
            for i = 1 : 2 : nargin
                switch varargin{ i }
                    case 'mode'
                        iMode = i + 1;
                    case 'tag'
                        iTag = i + 1;
                    otherwise
                        error( 'Invalid property' )
                end
            end

            % Set caller-defined values
            if ~isempty( iMode )
                if ~isrowstr( varargin{ iMode } )
                    error( 'Mode property must be a character string' )
                end
                obj.mode = varargin{ iMode };
            end
            if ~isempty( iTag )
                if ~ iscell( varargin{ iTag } ) ...
                   || ~isrowstr( varargin{ iTag } ) 
                    error( 'Invalid object tag' )
                end
                obj.tag = varargin{ iTag };
            end
        end
    end


    methods( Abstract )

        %% EVALUATESCALING  Evaluate distance
        y = evaluateScaling( obj )
        
        %% GETDEFAULTTAG Default tag for class
        tg = getDefaultTag( obj )

        %% IMPORTDATA  Get scaling data from components 
        D = importData( obj, comp )
    end

end    
