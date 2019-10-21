classdef nlsaDiffusionOperator_gl_mb_bs < nlsaDiffusionOperator_gl_mb_svd
%NLSADIFFUSIONOPERATOR_GL_MB_BS Class definition and constructor of diffusion 
% operator with multiple bandwidths and singular value decomposition instead
% of eigendecomposition. 
%
% This class implements the bistochastic kernels introduced in Coifman & Hirn
% (2013), "Bi-stochastic kernels via asymmetric affinity functions", Appl. 
% Comput. Harmon. Anal., 35, doi:177-180,10.1016/j.acha.2013.01.001. 
% 
% Modified 2018/06/15   

    methods

        %% CLASS CONSTRUCTOR
        function obj = nlsaDiffusionOperator_gl_mb_bs( varargin )

            obj = obj@nlsaDiffusionOperator_gl_mb_svd( varargin{ : } );

        end
    end

end    
