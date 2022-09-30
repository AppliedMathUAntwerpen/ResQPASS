function [ x_k, working_set_old, it, lam, convFlag] = qpas_schur(G,c,A,b,x0,working_set,Q,R,maxiter)

%    Solve a quadratic problem with the Active Set algorithm.
%    See Numerical Optimization (Nocedal and Wright) page 472.
%
%
%     The quadratic problem formulation is:
%     
%         minimize    (1/2)*x'*G*x + c'*x
%         subject to  A*x <= b
%         
%     The active set algorithm requires an initial guess x0 that
%     is feasible; i.e. A*x0 <= b.
%     
%     Parameters
%     ----------
%     G : Symmetric positive definite matrix, size = (n, n)
%     c : Vector, size = (n,1)
%     A : Matrix defining the linear inequality constraints.
%         shape = (m, n)
%     b : Vector of linear inequality constraint upper bounds.
%         shape = (m,1)
%     x0: Vector with an initial guess of the solution.
%         Must be feasible. size = (n,1)
%     working_set: a subset of active constraints of x0 (can be empty)
%         
%     Returns
%     -------
%     x : The solution of the QP.
%     working_set: The active set at the solution
%     it: number of iterations 

if nargin < 9
    maxiter = 5;
end

convFlag = false;

linsolveOpts.UT = true;

m = size(A, 1); x_k = x0; tol = 10^-6; 

L = chol(G)'; 

Ginv_c = (L')\(L\c); 

for it = 1:maxiter
%     it
    working_set_old = working_set;
    nact = length(working_set); 
     
%       Solve the KKT system for the equality constrained problem:
%       0.5 * p^T * G * p + (G * x_k + c)^T * p; s.t: A_i * p = 0
%       (for all rows i of A in the current working set)
%      
%       The resulting KKT system is:
%      [ G    A_i^T ] [ p   ]  = [ -(G * x_k + c) ]
%      [ A_i      0 ] [ lam ]    [              0 ] 

   if(nact == 0)
    p = -(x_k + Ginv_c); 
    lam = [];
    Ginv_A = [];
%     schur = [];
    Q = [];
    R = [];
    rhs = [];
   else
       if it == 1
           Ginv_A = linsolve(L',linsolve(L,(A(working_set,:)'), struct('LT', true)), linsolveOpts); % dit kan efficienter
%            Ginv_A = L'\(L\A(working_set,:)');
           schur = A(working_set,:)*Ginv_A;
           [Q,R] = qr(schur);
           rhs = A(working_set,:)*(x_k + Ginv_c);
       end
%     rhs = A(working_set,:)*(x_k + Ginv_c);

    lam = -linsolve(R,(Q'*rhs),linsolveOpts);
    p = -(x_k + Ginv_c +  Ginv_A*lam);
   end
   
   if(norm(p) > tol)
%   Find & add the blocking contraint to the working set
%   Find indices of all constraints not in the working set, where a_i^T * p > 0     
    Ap  = A*p;
    Ap_pos = find(Ap > 0);
    
    indices  = MY_intersect(MY_setdiff(1:m,working_set),Ap_pos);
    
    alphas = (b(indices) - A(indices,:)*x_k)./Ap(indices);
    
    [alpha,min_alpha_idx] = min(alphas);
    
    if(alpha<=1)
    bcidx = indices(min_alpha_idx);
    
    % Add blocking constraint to working set
    working_set = [working_set,bcidx];
    Ginv_A = [Ginv_A, (L')\(L\A(bcidx,:)')];
    column = A(working_set(1:end-1), :)*Ginv_A(:,end);
    row = A(bcidx, :)*Ginv_A;
    [Q,R] = qrinsert(Q,R,size(R,2)+1,column,'col');
    [Q,R] = qrinsert(Q,R,size(Q,1)+1,row,'row');
%     schur = [schur, column;
%              row];
%     [L_schur,U] = qr(schur);


%     U1 = L_schur\(A(working_set(1:end-1), :)*Ginv_A(:,end));
%     L1 = (A(bcidx, 1:end-1)*Ginv_A(1:end-1, 1:end-1))/U;
%     u0 = A(bcidx, :)*Ginv_A(:,end) - L1*U1;
%     L_schur = [L_schur,zeros(size(U1));L1,1];
%     U = [U,U1;zeros(size(L1)),u0];
       
    
    x_k = x_k + alpha*p;
    rhs = [rhs; A(bcidx,:)*(x_k + Ginv_c)];

    else

    x_k = x_k + p; 
        
    end
    
   else  % ( p = 0 )
       
      [min_lam,min_lam_idx] = min(lam);
      
%       If all Lagrange multipliers are positive, the solution is optimal.
%       Otherwise, the minimal Lagrange multiplier indicates the constraint
%       that must be removed from the working set.      

      if(min_lam < 0)
          working_set(min_lam_idx) = [];
          Ginv_A(:,min_lam_idx) = [];
%           schur(min_lam_idx,:) = [];
%           schur(:,min_lam_idx) = [];
%           [L_schur,U] = qr(schur);
          [Q,R] = qrdelete(Q,R,min_lam_idx,'col');
          [Q,R] = qrdelete(Q,R,min_lam_idx,'row');

%           L_schur(min_lam_idx,:) = [];
%           L_schur(:,min_lam_idx) = [];
%           U(min_lam_idx,:) = [];
%           U(:,min_lam_idx) = [];
          
%           schur = L_schur*U;
          
          rhs(min_lam_idx) = [];
          
      else % Solution found
%           disp('converged')
          convFlag = true;
          return          
      end
      
   end
   

end
%    disp('did not converge')
end


%https://nl.mathworks.com/matlabcentral/answers/53796-speed-up-intersect-setdiff-functions

function C = MY_intersect(A,B)
 P = zeros(1, max(max(A),max(B)) ) ;
 P(A) = 1;
 C = B(logical(P(B)));
end

function Z = MY_setdiff(X,Y)
  check = false(1, max(max(X), max(Y)));
  check(X) = true;
  check(Y) = false;
  Z = X(check(X));  
end
