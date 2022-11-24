function [y,V,x,obj,res,WS,nIters,LAM,MU] = subspace_qpas_restarted_krylov_functie(A,b,l,u, maxInnerIt, warmStart)

if nargin<5
    maxInnerIt = 100;
    warmStart = true;
elseif nargin<6
    warmStart = true;
end

M = size(A,1);
N = size(A,2);

l0 = l;
u0 = u;
x = zeros(N,1);
ws = [];

outer=1;

nIters = [];

% for outer=1:1
    y=[];
    r = b-A*x;
    l = l0-x;
    u = u0-x;
    v = A'*r;
    if outer==1
        V = [v/norm(v)];
    else
        i1 = find(lam<0);
        i2 = find(mu<0);
        V = [V(:,end)];
    end
    maxit = min(M,N);
    AV = A*V;
    H=[];
    for it=1:maxit
%         it
        if ~warmStart
            ws = [];
        end

        H(it,it) = AV(:,it)'*AV(:,it);
        if (it>1)
            H(1:it-1,it) = AV(:,1:it-1)'*AV(:,it);
            H(it,1:it-1) = AV(:,it)'*AV(:,1:it-1);
        end
        f = -r'*AV;

        try
        [y, ws, innerIters, lagMult] = qpas_schur(H,f',[V;-V],[u,-l],[y;0],ws,[],[],maxInnerIt);
        catch
            maxit = it;
            warning("Stopped because matrix was not positive definite")
            x  = x + V(:,1:it-1)*y;
%             size(V)
%             size(A)
%             figure; heatmap(abs(V(:,1:end-1)'*A'*A*V(:,1:end-1)), GridVisible="off");
            break;
        end
        nIters(it) = innerIters;
        LAMBDA = zeros(2*N,1);
        LAMBDA(ws) = lagMult;
        lam = LAMBDA(1:N);
        mu  = LAMBDA(N+1:end);
        LAM(:,it) = lam;
        MU(:,it) = mu;
        WS{it} = ws;

        obj((outer-1)*maxit+it) = (AV*y-r)'*(AV*y-r);

        v = A'*(AV*y-r) + lam -mu;
        V = [V v/norm(v)];
        res((outer-1)*maxit+it) = norm(v);
        
        if norm(v) < 1e-12
%             nIters = it-1;
            x  = x + V(:,1:it-1)*y;
            return
        end

        %Update AV
        AV = [AV,A*V(:,end)];

    end
%     size(V)
%     size(y)
%     x  = x + V(:,1:it-1)*y;
%     nIters = maxit;
    
% end

end