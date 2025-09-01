function xENorm = getIsrStateErrorNorm( obj, varargin )
% GETISRSTATEERRORNORM Get ISR state error norm of an nlsaModel_error object
%
% Modified 2014/07/24

xENorm = getDataNorm( getIsrErrComponent( obj ), varargin{ : } );
