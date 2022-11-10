function [y,V,x,obj,res,WS,nIters] = subspace_qpas_restarted_krylov_functie(A,b,l,u)

M = size(A,1);
N = size(A,2);

l0 = l;
u0 = u;
x = zeros(N,1);
ws = [];

outer=1;
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
    maxit = 300;
    for it=1:maxit
%         it
        AV = A*V;
        H = AV'*AV;
        f = -r'*A*V;

        try
        [y, ws, ~, lagMult] = qpas_schur(H,f',[V;-V],[u,-l],[y;0],ws,[],[],100);
        catch
            maxit = it;
            warning("Stopped because matrix was not positive definite")
%             x  = x + V(:,1:it-1)*y;
            nIters = it-1;
            break;
        end
        LAMBDA = zeros(2*N,1);
        LAMBDA(ws) = lagMult;
        lam = LAMBDA(1:N);
        mu  = LAMBDA(N+1:end);
%         LAM(:,it) = lam;
%         MU(:,it) = mu;
        WS{it} = ws;

        obj((outer-1)*maxit+it) = (A*V*y-r)'*(A*V*y-r);

        v = A'*(A*V*y-r) + lam -mu;
        V = [V v/norm(v)];
        res((outer-1)*maxit+it) = norm(v);
    end
    x  = x + V(:,1:maxit-1)*y;
    
% end

end