function [x,y,V,res,iters,obj,recursiveError,X] = ResQPASSv2(A,b,l,u, maxInnerIt, M1, maxOuterIt, doWarmStart, doRecursiveQPAS, resTol)
%ResQPASS Residual Quadratic Programming Active Set Subspace algorithm.
%
%   x = ResQPASS(A,b,l,u) solves the box-constrained linear least-squares
%   problem:
%       min  || A*x - b ||.
%      l<x<u 
%   If l or u are not provided, they will be set to Inf (unconstrained)
%   
%   x = ResQPASS(A,b,l,u,maxInnerIt) limits the number of inner QPAS
%   iterations. (default: 10)
%
%   x = ResQPASS(A,b,l,u,maxInnerIt,M1) preconditions the method with the
%   function M1. If M1 is empty, no preconditioning happens. (default: [])
%   example: M1 = @(x) U\(L\x) with [L,U]=ilu(A'*A).
%
%   x = ResQPASS(A,b,l,u,maxInnerIt,M1,maxOuterIt) limits the number of outer
%   iterations. (default: min(size(A)))
%
%   x = ResQPASS(A,b,l,u,maxInnerIt,M1,maxOuterIt,doWarmStart) turns warm
%   starting of the QPAS subroutine on or off. (default: true)
%
%   x = ResQPASS(A,b,l,u,maxInnerIt,M1,maxOuterIt,doWarmStart,doRecursiveQPAS) 
%   turns recursive calculations in the QPAS subroutine on or off.
%   (default: true)
%
%   x = ResQPASS(A,b,l,u,maxInnerIt,M1,maxOuterIt,doWarmStart,doRecursiveQPAS, resTol) 
%   changes the tollerance for the residual to resTol. (default: 1e-8)
%
%   [x,y,V] = ResQPASS(___) also returns the basis of the subspace V and
%   the projected solution y. x = y*V.
%   
%   [x,y,V,res] = ResQPASS(___) returns the norm of the (generalized)
%   residual from each iteration.
%
%   [x,y,V,res,iters] = ResQPASS(___) returns the number of QPAS iterations
%   performed every outer iteration.
%
%   [x,y,V,res,iters,obj] = ResQPASS(___) returns the objective ||A*x-b||
%   for every iteration.
%
%   [x,y,V,res,iters,obj,recursiveError] = ResQPASS(___) returns the
%   recursion error for every inner QPAS iteration.
%
%   [x,y,V,res,iters,obj,recursiveError,X] = ResQPASS(___) returns the
%   approximatex solution xk for every outer iteration k as columns of X.

%%%%%%%%%%%%%%%%%%%%%%%%%%
% Default input handling %
%%%%%%%%%%%%%%%%%%%%%%%%%%
EPSres = 1e-8;      % Default tolerance of the residual
EPSchol = 1e-10;    % Default tolerance of cholesky factorisation

if nargin < 2
    error('Not enough input arguments.')
end
[M,N] = size(A);            % Size of the problem

if nargin == 2
    l = -Inf*ones(N,1);
    u = Inf*ones(N,1);
    maxInnerIt = 10;
    doPreconditioning = false;
    maxOuterIt = min(M,N);
    doWarmStart = true;
    doRecursiveQPAS = true;
elseif nargin == 3
    u = Inf*ones(N,1);
    if isempty(l)
        l = -Inf*ones(N,1);
    end
    maxInnerIt = 10;
    doPreconditioning = false;
    maxOuterIt = min(M,N);
    doWarmStart = true;
    doRecursiveQPAS = true;
elseif nargin == 4
    if isempty(l)
        l = -Inf*ones(N,1);
    end
    if isempty(u)
        u = Inf*ones(N,1);
    end
    maxInnerIt = 10;
    doPreconditioning = false;
    maxOuterIt = min(M,N);
    doWarmStart = true;
    doRecursiveQPAS = true;
elseif nargin == 5
    doPreconditioning = false;
    maxOuterIt = min(M,N);
    doWarmStart = true;
    doRecursiveQPAS = true;
elseif nargin == 6
    if isempty(M1)
        doPreconditioning = false;
    else
        doPreconditioning = true;
    end
    maxOuterIt = min(M,N);
    doWarmStart = true;
    doRecursiveQPAS = true;
elseif nargin == 7
    if isempty(M1)
        doPreconditioning = false;
    else
        doPreconditioning = true;
    end
    doWarmStart = true;
    doRecursiveQPAS = true;
elseif nargin == 8
    if isempty(M1)
        doPreconditioning = false;
    else
        doPreconditioning = true;
    end
    doRecursiveQPAS = true;
elseif nargin == 9
    if isempty(M1)
        doPreconditioning = false;
    else
        doPreconditioning = true;
    end
elseif nargin == 10
    if isempty(M1)
        doPreconditioning = false;
    else
        doPreconditioning = true;
    end
    EPSres = resTol;
elseif nargin > 10
    error('Too many input arguments.')
end

if nargout > 6  % Initialise the recursion error only when requested
    recursiveError = [];
end

%%%%%%%%%%%%%%%%%%
% Initialisation %
%%%%%%%%%%%%%%%%%%
v = A'*b;                   % Initial residual
V = v/norm(v);              % Initial basis
AV = A*V;                   % Helper variable
L = sqrt(AV'*AV);           % Lower cholesky factor of hessian V'A'AV (=LL')
f = -b'*AV;                 % Linear factor
workingSet = []; y = [];    % Working set for QPAS
lu = [-l,u];                % Right hand side for qpas inequality
VV = [-V;V];                % Inequality matrix for qpas TODO: REMOVE THE USE OF V SOMEHOW?

%Prealocation for speed
res = zeros(maxOuterIt,1);

%%%%%%%%%%%%%%
% Outer loop %
%%%%%%%%%%%%%%
normRes = [];
stepType = [];
for outerIt = 1:maxOuterIt
    %%%%%%%%%%%%%%%%%%%%
    % Inner loop (QPAS)%
    %%%%%%%%%%%%%%%%%%%%
    if ~doWarmStart         % Reset working set when cold-starting
        workingSet = [];
    end
    if nargout <= 6         % Standard QPAS
        [y,workingSet,lagMultActive,iters(outerIt)] = ...
            qpasCholeskyv2(L,f,VV,lu,[y;0],workingSet,maxInnerIt,doRecursiveQPAS);
    else                    % QPAS with computation of recursion error
        [y,workingSet,lagMultActive,iters(outerIt),recursiveErrorIter] = ...
            qpasCholeskyv2(L,f,VV,lu,[y;0],workingSet,maxInnerIt,doRecursiveQPAS);
        recursiveError = [recursiveError, recursiveErrorIter];
    end
    % Lagrange multipliers
    lagMult = zeros(2*N,1);
    lagMult(workingSet) = lagMultActive;
    lambda = lagMult(1:N);          % Lower bounds
    mu = lagMult(N+1:end);          % Upper bounds
    
    if nargout > 7      % Approximate solution every iteration
        X(:,outerIt) = V*y;
    end

    %%%%%%%%%%%%
    % Residual %
    %%%%%%%%%%%%
    temp = (AV*y-b);                % Used for objective and residual
    if nargout > 5                  % compute objective
        obj(outerIt) = temp'*temp;
    end
    v = A'*temp - lambda + mu;
    if doPreconditioning
        v = M1(v);
    end
    res(outerIt) = norm(v);


    %%%%%%%%%%%%
    % Stopping %
    %%%%%%%%%%%%
    if res(outerIt) < EPSres        % Small residual
        x = V*y;
        res = res(1:outerIt);
        return
    elseif L(end,end) < EPSchol     % Breakdown of choleksy (happy breakdown)
        x = V*y;
        res = res(1:outerIt);
        return
    end
    
    %%%%%%%%%%%
    % Updates %
    %%%%%%%%%%%
    V = [V v/res(outerIt)];     % Basis expansion
    VV = [VV, [-V(:,end);V(:,end)]];                % Inequality matrix for qpas TODO: REMOVE THE USE OF V SOMEHOW?
    AV = [AV,A*V(:,end)]; %TODO: we can do it without (through L only)?
    U12 = L\(AV(:,1:outerIt)'*AV(:,outerIt+1));
    L = [L, zeros(outerIt,1);
            U12',sqrt(AV(:,outerIt+1)'*AV(:,outerIt+1)-U12'*U12)];
    f = [f;-b'*AV(:,end)];      % Linear term
end

x= V(:,1:end-1)*y;
warning("ResQPASS did not converge in %i iterations", maxOuterIt)

end