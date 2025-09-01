function [ xiENorm, xiNorm ] = getIsrVelocityErrorNorm( obj, varargin )
% GETISRVELOCITYERRORNORM Get ISR velocity error norm of an nlsaModel_err object
%
% Modified 2014/07/24

[ xiENorm, xiNorm ] = getVelocityNorm( getIsrErrComponent( obj ), varargin{ : } );
