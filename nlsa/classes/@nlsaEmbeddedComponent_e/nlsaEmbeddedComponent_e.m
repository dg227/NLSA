classdef nlsaEmbeddedComponent_e < nlsaEmbeddedComponent
%NLSAEMBEDDEDCOMPONENT_E  Class definition and constructor of NLSA 
% time-lagged embedded component with explicit storage of embedded data
%
% Modified 2014/04/04    

    methods

        %% CLASS CONSTRUCTOR
        function obj = nlsaEmbeddedComponent_e( varargin )
            obj@nlsaEmbeddedComponent( varargin{ : } );
        end
   end
end    
