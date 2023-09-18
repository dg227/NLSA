classdef nlsaDiffusionOperator_gl_mb_bs_pf < nlsaDiffusionOperator_gl_mb_bs
%NLSADIFFUSIONOPERATOR_GL_MB_BS_PF Class definition and constructor of diffusion 
% operator with bistochastic kernel normalization and positive-frequency
% projection.
%
% 
% Modified 2023/07/31

    methods

        %% CLASS CONSTRUCTOR
        function obj = nlsaDiffusionOperator_gl_mb_bs_pf(varargin)
            obj = obj@nlsaDiffusionOperator_gl_mb_bs(varargin{:});
        end

    end

end    
