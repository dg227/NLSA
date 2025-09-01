classdef nlsaLocalDistance
%NLSALOCALDISTANCE  Class definition and constructor of local pairwise distance 
% used in kernels
%
% explicit -> form data vectors in lagged embedding space and compute distances
% implicit -> compute distances in data space and sum over embedding indices
%
% Modified 2014/05/12   

   %% PROPERTIES
    properties
        tag  = '';
        mode = 'explicit'; 
    end

    methods

        %% NLSALOCALDISTANCE  Class contructor
        function obj = nlsaLocalDistance( varargin )

            % Parse input arguments
            iTag  = [];
            iMode = [];
            for i = 1 : 2 : nargin
                switch varargin{ i }
                    case 'tag'
                        iTag = i + 1;
                    case 'mode'
                        iMode = i + 1;
                    otherwise
                        error( 'Invalid property' )
                end
            end

            % Set caller-defined values
            if ~isempty( iTag )
                if ~iscell( varargin{ iTag } ) ...
                  && ~isrowstr( varargin{ iTag } ) 
                    error( 'Invalid object tag' )
                end
                obj.tag = varargin{ iTag };
            end
            if ~isempty( iMode )
                if ~isrowstr( varargin{ iMode } )
                    error( 'Mode property must be a character string' )
                end
                obj.mode = varargin{ iMode };
            end
        end
    end


    methods( Abstract )

        %% EVALDIST  Evaluate distance
        y = evaluateDistance( obj )
        
        %% GETDEFAULTTAG Default tag for class
        tg = getDefaultTag( obj )

        %% IMPORTDATA  Get distance data from components 
        D = importData( obj, comp )
    end

end    
