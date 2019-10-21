function [u,v] = XYmerge (x, y, EXY)
% Merge co-linear plot vectors
% This routine compresses an array of plot vectors by merging
% co-linear vectors.
%  x - vector of X coordinates
%  y - vector of Y coordinates
%  EXY - two element vector with the allowable error in X and Y
%        allowed when merging nearly co-linear segments. If
%        this parameter is missing, an error of 1/10 of a point
%        (maximum error 1/720 of an inch) is allowed. The
%        corresponding error in data units is based on the
%        current axis limits and axis sizes.
%
% Notes on Setting EXY:
% - The default axes size in Matlab is 434 by 342 pixels.
%   Using a conversion factor of 96 pixels per inch, this is
%   4.52 inches by 3.56 inches. Call these dimensions
%   Xsize and Ysize. The routine SetPlotSize can be used to
%   set the axes size to other values.
% - We will assume that the graph is to be plotted will be
%   imported into a document at full size.
% - Assume that the printout will occur on a printer with
%   resolution DPI (dots per inch). For instance a standard
%   laser printer resolution is 600 dpi.
% - The allowable error will be Edot (measured in printer
%   dots).
% - Let the data range in the X and Y directions be
%   [Xmin, Xmax] and [Ymin, Ymax]. Note that we may want to
%   print only part of the data range to "zoom" in on details.
%   If the whole data range is to be plotted, these values can
%   be found using the min and max operations.
% - Lets focus on the X direction. The total extent of the
%   plot in dots is Xsize * DPI. The allowable relative
%   error is Edot / (Xsize * DPI). In data units this is
%   (Xmax-Xmin) * Edot / (Xsize * DPI).
% - Assume an error of Edot=1 is acceptable and using the
%   default Xsize=4.52 inches and DPI = 600. Then the
%   X component of the error should be set to
%     EXY(1) = (Xmax-Xmin) / 2712.
% - Even easier, let this routine determine a reasonable
%   value for the acceptable error.
%   (a) Set the plot size (SetPlotSize). Choose the plot size
%       here so that it can be included in a document with no
%       further scaling.
%   (b) Call axis(Xmin,Xmax,Ymin,YMax) to set the plot limits.
%   (c) Call XYmerge to merge line segments (omitting the EXY
%       argument.
%   (d) Plot the data
%   (e) Set the plot limits again (plot wiped out the settings).
%   (f) Write the plot to a file (using WritePlot to ensure
%       no scaling of the plot).
%   (g) Include the plot into a document - don't scale the plot
%       while including it.

% $Id: XYmerge.m,v 1.14 2008/06/02 12:47:24 pkabal Exp $

% Notes:
% - The error criterion (when EXY is specified) is absolute
%   error. When used with data that is to be plotted on a
%   log scale, the linear data should converted to log before
%   being input to this routine. The returned data can then
%   be taken back to the linear domain for plotting.
% - The algorithm works by processing 3 points at a time.
%   Consider a line joining the end points of the triplet. The
%   length of the perpendicular distance from the middle point
%   to that line determines whether the middle point can be
%   omitted or not. The perpendicular distance has components
%   in the X and Y directions, and as such, even for equi-
%   spaced data, some tolerance should be allowed on the
%   X-error.
% - The algorithm is greedy but short-sighted. When it hits
%   a point at which the error in the line exceeds the
%   tolerance, it does not try to extend the line beyond
%   that point, when in fact there may be a viable longer
%   line.

N = length(x);
if (N ~= length(y))
  error ('XYmerge: Unequal length vectors');
end

if (nargin < 3)
  [DataPt, XScale, YScale] = SFdataXpt; % Data units per point
  ExA = 0.1 * DataPt(1);
  EyA = 0.1 * DataPt(2);
else
  XScale = 'linear';
  YScale = 'linear';
  ExA = EXY(1);
  EyA = EXY(2);
end

LogX = strcmp(XScale, 'log');
LogY = strcmp(YScale, 'log');
if (LogX)
  x = log10(x);
end
if (LogY)
  y = log10(y);
end

B = 1;
k = 0;
k = k + 1;
kv(k) = B;
while (B <= N-1)

  % B - is a base point
  % G - is a known good point (the line from B to G can be
  %     approximated by a single line)
  % T - is a test point beyond G - we will test the line from
  %     B to T and reset G to T if we are successful.
  % Infinite or NaN values for B, G and/or T result in OK = 0

  G = B + 1;
  step = 1;
  straddle = 0;
  Lx = length(x);
  while (1)
    T = G + step;
    if (T > Lx)   % Don't look beyond the array
      straddle = 1;
      OK = 0;
    else

      % Main test loop (loop over B+1:T-1, permuted)
      % The indices are arranged from the middle out to abort
      % the test as quickly as possible - Tests indicate a 25%
      % reduction in evaluations of Test3 over a sequential
      % search
      I = B+1:T-1;
      Ni = length(I);
      II = reshape([I; fliplr(I)], 1, []);
      II = II(Ni+1:end);   % Search from middle out
      for (i = II)

	    % Catch inflection points where the line changes from or to
        % an exact horizontal or an exact vertical
	    OK = TestVH(x(B), y(B), x(i), y(i), x(T), y(T));

	    if (isnan(OK))
          OK = Test3(x(B), y(B), x(i), y(i), x(T), y(T), ExA, EyA);
	    end
        if (~ OK)
          straddle = 1;
          break
        end
      end

    end

    %  (1) If we have found a T which does not work, we know
    %      the end point is in the interval G:T-1. Use a
    %      binary search (decreasing the step size) to find the
    %        end point.
    %  (2) If we have not found a T which does not work, keep
    %      looking by increasing the step size.
    if (straddle)
      if (OK)
        G = T;
      end
      if (step == 1)
        break
      end
      step = step / 2;
      continue
    else    % success, but not straddling the end point
      G = T;
      step = 2 * step;
      continue
    end
  end

  B = G;
  k = k + 1;
  kv(k) = B;
end

if (LogX)
  u = 10.^x(kv);
else
  u = x(kv);
end
if (LogY)
  v = 10.^y(kv);
else
  v = y(kv);
end

fprintf('XYMerge: No. points (in/out): %d/%d\n', N, k);

return

%==========
function OK = Test3 (x1, y1, x2, y2, x3, y3, ExA, EyA)

% Consider 3 points.  Draw a straight line between the end
% points. The middle point will be skipped if the X and Y
% components of the perpendicular distance from the middle
% point to the straight line are smaller than given
% tolerances.
%
% The first step is to translate the axes so that (x(m),y(m))
% becomes (0,0).
%
%               [x2,y2]                  [p1,q1]
%                  o                          o
%                 / \                        / \
%                /   \                      /   \
%               /     o                    /     o
%              /   [x3,y3]                /   [p2,q2]
%     [x1,y1] o                    [0,0] o

% A rotation about the origin through an angle w (CCW)
% can be expressed as
%   [r] = [cos(w) -sin(w)] [p]
%   [s]   [sin(w)  cos(w)] [q].
% The perpendicular error at (p1,q1) can be determined by a
% rotation about (0,0) such that the line from the (0,0) to
% (p2,q2) is horizontal.  The rotation is through the angle -a,
% where
%   cos(a) = p2 / D,
%   sin(a) = q2 / D,
% and D = sqrt(p2^2 + q2^2).  The perpendicular error is the
% ordinate value of (p1,q1) after rotation,
%   errN = q1 cos(a) - p1 sin(a).
% The components of this error in the original X and Y
% directions can be found by projecting errN,
%   errX = sin(a) errN,
%   errY = -cos(a) errN.

% The test for the absolute X error is
%   |errX| > ExA.
% Or,
%   |sin(a) (q1 cos(a) - p1 sin(a))| > ExA
%
%    q2     p2      q2
%   |-- (q1 -- - p1 --)| > ExA
%    D      D       D
% or
%   |q2 (q1 p2 - p1 q2)| > D^2 ExA
% This rearrangement avoids possible divisions by zero.
%
% Similarly the check for the Y error is,
%   |p2 (q1 p2 - p1 q2)| > D^2 EyA

% The error region is a rectangle. It would be easy to change
% the error region to be an ellipse with the given errors
% as the axes.

% Any NaN in these expressions will result in OK = 0
p1 = x2 - x1;
q1 = y2 - y1;
p2 = x3 - x1;
q2 = y3 - y1;

D2 = p2^2 + q2^2;
err = q1 * p2 - p1 * q2;
OK = (abs(q2 * err) <= D2 * ExA) & (abs(p2 * err) <= D2 * EyA);

return

%==========
function OK = TestVH (x1, y1, x2, y2, x3, y3)

T12 = 0;
if (x1 == x2)
  T12 = 1;
elseif (y1 == y2)
  T12 = 2;
end

T23 = 0;
if (x2 == x3)
  T23 = 1;
elseif (y2 == y3)
  T23 = 2;
end

if (T12 == 0 && T23 == 0)
  OK = NaN;
elseif (T12 == T23)
  OK = 1;
else
  OK = 0;
end

return
