function x = SQPAS(A,b,delta)

outer_it = 4;
inner_it = 5;

% Height of the matrix (#rows)
M = size(A,1);
% Width of the matrix (#columns)
N = size(A,2);


u = delta*ones(N,1);
l = -u;

xbar = zeros(N,1);

ubar = u-xbar;
lbar = l-xbar;

r0 = -A'*b;
r=r0;

V = r/norm(r);
r = V;

% Initialisation
AV = zeros(M,inner_it);
H = zeros(inner_it);
ws=[];

res = zeros(1000,1);

% D = speye(N,N);

% L = ichol(sparse(A'*A), struct('droptol', 0.1));
% L = chol(A'*A);
% fig = figure('Position', [1286,402,829,361], 'Renderer', 'painters');

btilde = b;



for outer = 1:outer_it
for it = 1:inner_it
%     it = inner + 100*(outer-1);
    AV(:,it) = A*r;
    H(it,it) = AV(:,it)'*AV(:,it);
        if (it>1)
            H(1:it-1,it) = AV(:,1:it-1)'*AV(:,it);
            H(it,1:it-1) = AV(:,it)'*AV(:,1:it-1);
        end
    
      
    f = -btilde'*AV(:,1:it);
    if it==1
        [y, ws, nIters, lagMult, ~] = qpas_schur(H(1:it,1:it),f',[-V;V],[-lbar; ubar], 0, [],[],[],5);
%         maxit = nIters;
            % The unbounded solution
            y = AV(:,1)'*btilde/H(1,1);
            
%             % Equal to the bounded solution -> all lagrange multipliers = 0
%             if all([V;-V]*y <= delta)
%                 ws = [];
%                 lagMult = [];
%             % There is an active constraint -> the smalles one (univariate)
%             else
%                 [Min, ws] = min([V;-V]);
%                 y = delta/Min;
%                 lagMult = (H(1,1)*y - AV(:,1)'*btilde)/(-Min);
%             end

    else
        [y, ws, nIters, lagMult, ~] = qpas_schur(H(1:it,1:it),f',[-V;V],[-lbar; ubar], [y;0], ws,[],[],5);
%         maxit = max(maxit,nIters);
    end
    LAMBDA = zeros(2*N,1);
    LAMBDA(ws) = lagMult;
    lam = LAMBDA(1:N);
    mu  = LAMBDA(N+1:end);
    
    DEL = ones(N,1);
    DEL(ws(ws<=N)) = lbar(ws(ws<=N));
    DEL(ws(ws>N)-N) = ubar(ws(ws>N)-N);

    x = xbar + V*y;

%     r = A'*(A*V*y-b) + 1/delta*diag(lam +mu)*V*y;
%     r = A'*(A*x_approx);
%     r = r+r0;
%     D = spdiags(lam + mu,0,D);
%     r = r + deltaInv*D*x_approx;
        
%     r = AV(:,1:it)*y;
%     r = A'*r;
%     r = r+r0;
%     r = r + (mu-lam)./DEL.*x;

    r = A'*(A*x-btilde) + (mu-lam)./DEL.*x;

%     obj(it + inner_it*(outer-1)) = (A*x-b)'*(A*x-b);
    
    V = [V r/norm(r)];
    res(it + inner_it*(outer-1)) = norm(r);
    
    r = r/norm(r);
    
%     subplot(1,2,1)
%     imshow(reshape(rescale(x), sqrt(N),sqrt(N))', 'InitialMagnification', 800);
% %     imshow(reshape(x + 0.5, sqrt(N),sqrt(N)), 'InitialMagnification', 800);
% 
%     subplot(2,2,2)
% %     semilogy(abs(obj-exact_obj))        
%     semilogy(obj)
%     ax = gca;
%     ax.XTick = 100:100:700;
%     ax.YTick = [0.01,1,100];
%     ax.FontSize = 10;
%     legend("$f(V_ky_k)$", 'FontSize', 10)
%     xlim([1,750])
%     ylim([0.002, 100])
%     subplot(2,2,4)
%     semilogy(res)
%     ax = gca;
%     ax.XTick = 100:100:700;
%     ax.YTick = [0.01,1,100];
%     ax.FontSize = 10;
%     legend("residu", 'FontSize', 10)
%     xlabel("Iteratie $k$")
%     xlim([1,750])
%     ylim([0.002, 100])
%     sgtitle(strcat("Iteratie: ", num2str(it + inner_it*(outer-1))));
%     drawnow;
%         
    if it > 1
    if abs(res(it)-res(it-1)) < 1e-6
        return;
    end
    end
% 
%     if outer > 7
%         return;
%     end
    
end

xbar = x;
% ubar = u-xbar;
% lbar = l-xbar;

btilde = b-A*xbar;

AV = zeros(M,inner_it);
H = zeros(inner_it);
ws=[];

r = -A'*btilde;
V = r/norm(r);
r = V;

end
end

