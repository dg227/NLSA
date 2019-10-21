function xiRef = getOutReferenceVelocityNorm( obj, varargin )
% GETOSEREFERENCEVELOCITYNORM Get OS reference velocity norm of an 
% nlsaModel_err object
%
% Modified 2014/07/24

xiRef = getVelocityNorm( getOutRefComponent( obj ), varargin{ : } );
