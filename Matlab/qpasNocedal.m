function [ x_k ] = qpas(G,c,A,b,x0)

%    Solve a quadratic problem with the Active Set algorithm.
%     
%     The quadratic problem formulation is:
%     
%         minimize    (1/2)*x'*G*x + c'*x
%         subject to  A*x <= b
%         
%     The active set algorithm requires an initial guess x0 that
%     is feasible; ie A*x0 <= b.
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
%         
%     Returns
%     -------
%     x : The solution of the QP. 

[m,n] = size(A);

check = sum(A*x0 - b > 0);

if(check>0)
%     disp(['Initial guess is not feasible with respect to the constraint Ax<=b'])
    x_k = NaN;
    return
end

x_k = x0; 

% Initialize the working set
working_set = find(A*x0 == b)';

eps = 10^-8;
maxiter = 100; 

for it = 1:maxiter
    
%    disp(['-------------------------------------------------------'])
%    disp(['ITERATION ',num2str(it)])
%    disp(['-------------------------------------------------------'])
   
   nact = length(working_set); 
   
   x_k;
   working_set;
   
%       Solve the KKT system for the equality constrained problem:
%       0.5 * p^T * G * p + (G * x_k + c)^T * p; s.t: A_i * p = 0
%       (for all rows i of A in the current working set)
%      
%       The resulting KKT system is:
%      [ G    A_i^T ] [ p   ]  = [ -(G * x_k + c) ]
%      [ A_i      0 ] [ lam ]    [              0 ] 

   K = [G, A(working_set,:)'; A(working_set,:), zeros(nact,nact)];
   rhs = [-(G*x_k + c); zeros(nact,1)];

   plam = K\rhs;
   
   % split the solution from the Lagrange multipliers
   p = plam(1:n);
   lam = plam(n + 1:end); 
   
   if(norm(p) > eps)
%     disp('Current iterate x_k can be improved given the current working set')
%   Find & add the blocking contraint to the working set
%   Find indices of all constraints not in the
%   working set, where a_i^T * p > 0     
    Ap  = A*p;
    Ap_pos = find(Ap>0);
    
    mask  = intersect(setdiff(1:m,working_set),Ap_pos);
    
    alphas = (b(mask) - A(mask,:)*x_k)./Ap(mask);
    
    [alpha,min_alpha_idx] = min(alphas);
    
    if(alpha<=1)
    bcidx = mask(min_alpha_idx);
    
    % Add blocking constraint to working set
    working_set = [working_set,bcidx];
    
%     disp(['Blocking constraint ', num2str(bcidx),' found.'])
%     disp(['New iterate x_k = x_k + alpha p:'])
    x_k = x_k + alpha*p;
   
    
    else
        
%     disp(['No blocking contraint found.'])
%     disp(['New iterate x_k = x_k + p:'])
    x_k = x_k + p;
        
    end
    
   else    % ( p = 0 )
%       disp('No possible improvement to x_k given the current working set') 
      
      [min_lam,min_lam_idx] = min(lam);
      
%       If all Lagrange multipliers are positive, the solution is optimal.
%       Otherwise, the minimal Lagrange multiplier indicates the constraint
%       that must be removed from the working set.      

      if(min_lam < 0)
          rcidx = working_set(min_lam_idx);
          %working_set(working_set == rcidx) = []; % remove index from working set
          working_set(min_lam_idx) = [];
%           disp(['Removed index ', num2str(rcidx), ' from working set.'])
      else
          
%           disp(['All langrange multipliers are nonnegative, optimal solution found'])
          Solution = x_k;
          break
          
      end
      
   end
   
end

