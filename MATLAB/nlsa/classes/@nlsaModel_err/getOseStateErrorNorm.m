function xENorm = getOseStateErrorNorm( obj, varargin )
% GETOSESTATEERRORNORM Get OSE state error norm of an nlsaModel_error object
%
% Modified 2014/07/24

xENorm = getDataNorm( getOseErrComponent( obj ), varargin{ : } );
