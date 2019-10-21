function [ xiENorm, xiNorm ] = getOseVelocityErrorNorm( obj, varargin )
% GETOSEVELOCITYERRORNORM Get OSE velocity error norm of an nlsaModel_err object
%
% Modified 2014/07/24

[ xiENorm, xiNorm ] = getVelocityNorm( getOseErrComponent( obj ), varargin{ : } );
