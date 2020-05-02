classdef nlsaKoopmanOperator_rkhs < nlsaKoopmanOperator_diff
%NLSAKOOPMANOPERATOR_RKHS Class definition and constructor of Koopman generator
% with reproducing kernel Hilbert space (RKHS) compactification.
% 
% Modified 2020/04/08    

    %% METHODS
    methods

        %% CLASS CONSTRUCTOR
        function obj = nlsaKoopmanOperator_rkhs( varargin )

            obj = obj@nlsaKoopmanOperator_diff( varargin{ : } );
        end
    end

end    
