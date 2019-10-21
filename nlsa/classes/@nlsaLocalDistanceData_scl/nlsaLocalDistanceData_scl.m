classdef nlsaLocalDistanceData_scl < nlsaLocalDistanceData
%NLSALOCALDISTANCEDATA_SCL  Class definition and constructor of distance data 
% with scaling factors
%
% Modified 2015/10/28    

    %% PROPERTIES
    properties
        sclComponent = nlsaComponent();
    end

    methods
        %% NLSALOCALDISTANCEDATA_SCL  Class constructor
        function obj = nlsaLocalDistanceData_scl( varargin )

            nargin = numel( varargin );

            if nargin > 0 && isa( varargin{ 1 }, 'nlsaLocalDistanceData' )
                varargin = { 'component', getComponent( varargin{ 1 } ), ...
                             varargin{ 2 : end } };
            end
            nargin = numel( varargin );
            
            ifParentArg = true( 1, nargin );

            % Parse input arguments
            iSclComponent = [];
            for i = 1 : 2 : nargin
                switch varargin{ i }
                    case 'sclComponent'
                        iSclComponent = i + 1;
                        ifParentArg( [ i i + 1 ] ) = false;
                end
            end

            obj = obj@nlsaLocalDistanceData( varargin{ ifParentArg } );
           
            % Set caller-defined values
            if ~isempty( iSclComponent )
                if ~isa( varargin{ iSclComponent }, 'nlsaComponent' )
                    error( 'sclComponent must be set to an nlsaComponent object' )
                end
                nC = getNComponent( obj );
                if nC > numel( varargin{ iSclComponent } ) ...
                   && iscolumn( varargin{ iSclComponent } )
                    obj.sclComponent = repmat( varargin{ iSclComponent }, ...
                                               [ nC 1 ] );
                elseif all( size( obj.component ) == size( varargin{ iSclComponent } ) )
                    
                    obj.sclComponent = varargin{ iSclComponent };
                else
                    error( 'Incompatible scaling data' )
                end
            end
        end
    end
end
