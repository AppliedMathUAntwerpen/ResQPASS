%Use LaTeX font
set(groot,'defaulttextinterpreter','latex');  
set(groot, 'defaultAxesTickLabelInterpreter','latex');  
set(groot, 'defaultLegendInterpreter','latex');

%% Section 2
testActiveSetSize

%% Section 3
timingWarmStart
timingLimittingInner
timingRecursion

%% Section 4 (without ALS)
% Can take several minutes
timingLSQR

% Can take about an hour to run

% timingComparison
% testBoundedBalloon2D