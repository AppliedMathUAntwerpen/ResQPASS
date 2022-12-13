function [y,V,x,obj,res,WS,nIters,LAM,MU] = subspace_qpas_restarted_krylov_functie(A,b,l,u, maxInnerIt, warmStart)

%subspace_qpas_restarted_krylov_functie: Function that fascilitates easy
%testing of the sub-QPAS method.
%
%   This method solves the problem
%       $\min_x \|Ax-b\|_2^2$
%       $s.t.   l<=x<=u$
%
%   INPUT:
%       A: an MxN matrix, representation of the model
%       b: a vector of length M, the measured values
%       l: a vector of length N, lower bound of the solution
%       u: a vector of length N, upper bound of the solution
%       maxInnerIt: maximal number of inner QPAS iterations (default: 100)
%       warmStart: flag to enable/disable warm starting of QPAS (default: enabled)
%
%       l<=0<=u should hold to ensure feasibility of x=0 in the first
%       iteration (always possible through a shift)
%
%   OUTPUT:
%       y: optimal solution in the subspace
%       V: normalised basis vectors of the subspace
%       x: optimal solution in the original space (x=V*y)
%       obj: value of the objective funcion \|Ax-b\|_2^2
%       res: vector of the norms of the residuals
%       WS: struct of the working set after every (outer) iteration
%       nIters: vector with the number of inner qpas iterations every outer iteration
%       LAM: matrix with all lagrange multipliers (corresponding to the lower bound) every outer iteration
%       MU: matrix with all lagrange multipliers (corresponding to the upper bound) every outer iteration


% Initialise the default values if they are not set
if nargin<5
    maxInnerIt = 100;
    warmStart = true;
elseif nargin<6
    warmStart = true;
end

% Size of A
M = size(A,1);
N = size(A,2);

% Initialise
nIters = [];
y=[];
ws = [];
maxit = min(M,N);
f = []; % Coefficients of the linear part (0.5*x'Hx + f'x)

% Initial residual
v = -A'*b;
V = v/norm(v);
AV = A*V; %Helper variable

% Lower cholesky factor of the hessian V'A'AV (= LL')
L=sqrt(AV'*AV);

% Outer loop
for it=1:maxit
    % Reset the working set if the warm start flag is disabled
    if ~warmStart
        ws = [];
    end
    
    % Linear factor 
    f = [f,-b'*AV(:,end)];

    %QPAS solution (inner iteration)
    [y, ws, nIters(it), lagMult] = qpas_schur(L,f',[V;-V],[u,-l],[y;0],ws,[],[],maxInnerIt);

    % Extract the lambda en mu out of the lagrange multipliers given by qpas
    LAMBDA = zeros(2*N,1);
    LAMBDA(ws) = lagMult;
    lam = LAMBDA(1:N);
    mu  = LAMBDA(N+1:end);
    
    % Save the lagrange multipliers (only for analysis purpose)
    LAM(:,it) = lam;
    MU(:,it) = mu;
    
    % Save the working set (only for analysis purpose)
    WS{it} = ws;

    % Compute the current objective (only for analysis purpose)
    temp = (AV*y-b);
    obj(it) = temp'*temp;

    % Compute the residual (new basis vector)
    v = A'*temp + lam -mu;
    V = [V v/norm(v)];
    res(it) = norm(v);
    
    % Stop if the residual is sufficiently small
    if norm(v) < 1e-12
        nIters = it-1;
        x = V(:,1:it-1)*y;
        return
    end

    %Update AV
    AV = [AV,A*V(:,end)];

    % Update the Cholesky factorisation (1 row and 1 column added to the hessian)
    U12 = linsolve(L,AV(:,1:it)'*AV(:,it+1),struct('LT', true));
    L = [L, zeros(it,1);
            U12',sqrt(AV(:,it+1)'*AV(:,it+1)-U12'*U12)];
    
    % Stop if positive definiteness is lost (a (near) zero is added to the
    % diagonal of L, i.e. the hessian has a (near) zero eigenvalue)
    if prod(diag(L)) < 1e-10
        warning("Stopped because matrix was not positive definite (Cholesky factorisation does not exist)")
        x  = V(:,1:it)*y;
        return;
    end

end
end