function [H, f] = AdjustFreq (b, a,  Fspec, NPlot)
% [H, f] = AdjFreq (b, a, Fspec, NPlot)
%
% Adjust the frequencies of the frequency response evaluation points.
% The initial evaluation points are uniformly spaced between Fmin and
% Fmax (inclusive). The local maxima and minima of the magnitude of the
% frequency response are found. Note that these extrema include zeros of
% the response. The frequencies corresponding to these extrema are stepped
% up and stepped down (initially by half the frequency grid spacing).
% Those modified evaluation points which give higher maxima or lower
% minima are kept. The step size is halved and the process is repeated
% for a number of iterations.
%
% The frequencies for points which do not correspond to extrema are not
% changed. The frequencies for points which correspond to extrema are
% moved will always remain between the adjacent initial evaluation points.
%
% b - Numerator coefficients
% a - Denominator coefficients
% Fspec - Frequency axis specifications. This specification can be give
%    in a structure or a 3 element array. The 3 elements correspond to the
%    structure elements:
%      Fspec.Lim: Two element array - [Fmin, Fmax]
%      Fspec.Fs: Sampling frequency (assumed to be one if not defined)
% Nplot - Number of plot points. This number should be large enough so
%    that there is only one extremum of the response between the samples
%    of the fequency response.
%
% $Id: AdjustFreq.m,v 1.2 2008/05/01 18:53:04 pkabal Exp $

if (isstruct (Fspec))
  if (~ isfield(Fspec, 'Fs'))
    Fspec.Fs = 1;
  end
else
  Ftemp = Fspec;
  Fspec.Lim = Ftemp(1);
  Fspec.Fmax = Ftemp(2);
  if (length (Ftemp) < 3)
    Ftemp(3) = 1;
  end
  Fspec.Fs = Ftemp(3);
end

f = linspace (Fspec.Lim(1), Fspec.Lim(2), NPlot);
f = f(:);
H = freqz (b, a, f, Fspec.Fs);

% Find the local maxima and minima
Hdiff = diff (abs (H));
IExt = (Hdiff(1:end-1) .* Hdiff(2:end) < 0);
IMax = logical ([0; (Hdiff(1:end-1) > 0 & IExt); 0]);
IMin = logical ([0; (Hdiff(1:end-1) < 0 & IExt); 0]);

fstep = 0.5 * (f(NPlot) - f(1)) / (NPlot - 1);
IExt = (IMax | IMin);

NIter = 8;
for (i = 1:NIter)

  Ha = abs (H);

  % Increment and decrement the frequencies of the extrema
  fu = f + fstep * IExt;
  fd = f - fstep * IExt;

  % Evaluate the frequency response at the modified frequencies
  Hu = freqz (b, a, fu, Fspec.Fs);
  Hd = freqz (b, a, fd, Fspec.Fs);

  % Find the points for which the modified frequencies
  % give a better rendition of the maxima or minima
  Hua = abs (Hu);
  Hda = abs (Hd);
  SU = (((Hua > Ha) & IMax) | ((Hua < Ha) & IMin));
  SD = (((Hda > Ha) & IMax) | ((Hda < Ha) & IMin));

  f(SU) = fu(SU);
  H(SU) = Hu(SU);
  f(SD) = fd(SD);
  H(SD) = Hd(SD);

  fstep = 0.5 * fstep;

end

return

