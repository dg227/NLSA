classdef nlsaDiffusionOperator_ose < nlsaDiffusionOperator_batch
%NLSADIFFUSIONOPERATOR_OSE Class definition and constructor of diffusion 
%  operator with out of sample extension (OSE) via the Nystrom method
% 
% Modified 2016/01/2  
    methods

    % Bandwidth parameter epsilon for this class is a multiplier of 
    % the bandwidth for the in-sample data

        function obj = nlsaDiffusionOperator_ose( varargin )

            nargin = numel( varargin );

            if nargin == 1
                if isa( varargin{ 1 }, 'nlsaDiffusionOperator' )
                
                    varargin = { 'alpha',             getAlpha( varargin{ 1 } ), ...
                                 'epsilon',           getEpsilon( varargin{ 1 } ), ...
                                 'nEigenfunction',    getNEigenfunction( varargin{ 1 } ), ...
                                 'path', getPath( varargin{ 1 } ), ...
                                 'operatorSubpath', getOperatorSubpath( varargin{ 1 } ), ...
                                 'eigenfunctionSubpath', getEigenfunctionSubpath( varargin{ 1 } ), ...
                                 'eigenvalueFile', getEigenvalueFile( varargin{ 1 } ), ...
                                 'tag', getTag( varargin{ 1 } ) };

                    nargin = numel( varargin );
                end
            end

            obj = obj@nlsaDiffusionOperator_batch( varargin{ :  } );
        end
    end
end    
