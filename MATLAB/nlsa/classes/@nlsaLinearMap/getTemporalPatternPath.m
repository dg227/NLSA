function pth = getTemporalPatternPath( obj )
% GETTEMPORALPATTERNPATH Get temporal pattern path of an 
% nlsaLinearMap object 
%
% Modified 2015/10/19

pth = fullfile( getPath( obj ), getTemporalPatternSubpath( obj ) );
