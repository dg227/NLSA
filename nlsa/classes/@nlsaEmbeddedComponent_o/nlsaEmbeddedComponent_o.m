classdef nlsaEmbeddedComponent_o < nlsaEmbeddedComponent
%NLSAEMBEDDEDCOMPONENT_O Class definition and constructor of NLSA 
% time-lagged embedded component with implicit storage of embedded data using
% overlapping batches
%
% Modified 2014/05/14    

    methods

        %% CLASS CONSTRUCTOR
        function obj = nlsaEmbeddedComponent_o( varargin )

            obj = obj@nlsaEmbeddedComponent( varargin{ : } );
        end
    end
end    
