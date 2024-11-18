function w = hanning(varargin)
%HANNING   Hanning window.
%   HANNING(N) returns the N-point symmetric Hanning window in a column
%   vector.  Note that the first and last zero-weighted window samples
%   are not included.
%
%   HANNING(N,'symmetric') returns the same result as HANNING(N).
%
%   HANNING(N,'periodic') returns the N-point periodic Hanning window,
%   and includes the first zero-weighted window sample.
%
%   NOTE: Use the HANN function to get a Hanning window which has the 
%          first and last zero-weighted samples. 
%
%   See also BARTLETT, BLACKMAN, BOXCAR, CHEBWIN, HAMMING, HANN, KAISER
%   and TRIANG.

%   Copyright 1988-2020 The MathWorks, Inc.
%#codegen

% Check number of inputs
narginchk(1,2);

% Check for trivial order
[n,wt,trivialwin] = check_order(varargin{1});
if trivialwin 
    w = cast(wt,class(n));
    return
end

% Select the sampling option
if nargin == 1
   sflag = 'symmetric';
else
   sflag = validatestring(lower(varargin{2}),{'symmetric','periodic'},...
           'hanning','sflag',2);
end

% Evaluate the window
w = coder.nullcopy(zeros(n,1,class(n)));
switch sflag
case 'periodic'
   % Includes the first zero sample
   w = [0; sym_hanning(n-1)];
case 'symmetric'
   % Does not include the first and last zero sample
   w = sym_hanning(n);
end

%---------------------------------------------------------------------
function w = sym_hanning(n)
%SYM_HANNING   Symmetric Hanning window. 
%   SYM_HANNING Returns an exactly symmetric N point window by evaluating
%   the first half and then flipping the same samples over the other half.
coder.inline('always')
if iseven(n)
   % Even length window  
   w = calc_hanning(n/2,n);
   w = [w; w(end:-1:1)];
else
   % Odd length window
   w = calc_hanning((n+1)/2,n);
   w = [w; w(end-1:-1:1)];
end

%---------------------------------------------------------------------
function w = calc_hanning(m,n)
%CALC_HANNING   Calculates Hanning window samples.
%   CALC_HANNING Calculates and returns the first M points of an N point
%   Hanning window.
coder.inline('always')
w = .5*(1 - cos(2*pi*(1:m)'/(n+1))); 

% [EOF] hanning.m

% LocalWords:  sflag CALC
