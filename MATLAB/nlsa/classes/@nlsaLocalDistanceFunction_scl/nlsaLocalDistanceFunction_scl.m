classdef nlsaLocalDistanceFunction_scl < nlsaLocalDistanceFunction
%NLSALOCALDISTANCEFUNCTION_SCL  Class definition and constructor of local 
% distance function with scaling
%
% Modified 2015/10/31    

    %% PROPERTIES
    properties
        lScaling = nlsaLocalScaling_pwr();
        QS       = struct(); % query scaling data
        QT       = struct(); % query test data
    end

    methods
        %% NLSALOCALDISTANCEFUNCTION_SCL  Class constructor
        function obj = nlsaLocalDistanceFunction_scl( varargin )

            nargin = numel( varargin );

            if nargin > 0 && isa( varargin{ 1 }, 'nlsaLocalDistanceFunction' )
                varargin = { 'localdistance', getLocalDistance( varargin{ 1 } ), ...
                             'tag', getTag( varargin{ 1 } ), ...
                             varargin{ 2 : end } };
            end
            nargin = numel( varargin );
            
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

            obj = obj@nlsaLocalDistanceFunction( varargin{ ifParentArg } );
           
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
