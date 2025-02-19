clear all; 
N = 100;
e = ones(N,1);
h = 1/(N+1);
x = h*(1:N);
A = -(1/h^2)*spdiags([e -2*e e],[-1,0,1],N,N);

A2d = kron(A,speye(N)) + kron(speye(N),A);
A =A2d;
b = 2.0*ones(N,1);
b2d = kron(b,b);
b = b2d;
[X,Y] = meshgrid(x,x);
l = zeros(N*N,1);
u = 0.1*ones(N*N,1);

%% solution unpreconditioned
% H = A'*A;
% c =  -b'*A;
% tic
% [sol_ip] = quadprog(H,c,[],[],[],[],l,u);
% toc
% 
% obj_ip = norm(A*sol_ip-b)^2;

tic
[sol,y_unprec,V,res,iters,obj,~,X_unprec] = ResQPASSv2(A,b,l,u,10000,[],200);
toc


%%
options.type = 'crout';
options.milu = 'row';
options.droptol = 0.1;


[L, U] = ilu(A'*A,options);
M = @(x)    U\(L\ x);

tic
% [sol_prec,y_prec,V,res_prec,iters_prec,obj_prec,~,X_prec] = ResQPASSv2(A,b,l,u,10000,M,200);
[sol_prec,y_prec,V,res_prec,iters_prec,obj_prec] = ResQPASSv2(A,b,l,u,10000,M,200);
% [sol_prec] = ResQPASSv2(A,b,l,u,10000,M,200);
toc

%% Figures
% width = 17; height = 6;
% %LaTeX font
% set(groot,'defaulttextinterpreter','latex');  
% set(groot, 'defaultAxesTickLabelInterpreter','latex');  
% set(groot, 'defaultLegendInterpreter','latex');
% 
% markings = {'-o', '-square','-diamond','-^','-*','-pentagram'};
% colors = {'#1B9E77','#D95F02','#7570B3','#E7298A','#66A61E','#E6AB02'};
% 
% figure(Units="centimeters", Position=[4 4 width height], PaperUnits="centimeters", PaperSize=[width height]);
% subplot(1,2,1)
% mesh(X,Y,reshape(sol_prec,N,N))
% title(strcat("Solution preconditioned, ",num2str(length(y_prec)), " iterations"))
% 
% subplot(1,2,2)
% mesh(X,Y,reshape(sol,N,N))
% title(strcat("Solution unpreconditioned, ",num2str(length(y_unprec)), " iterations"))
% 
% % subplot(2,3,3)
% % mesh(X,Y,reshape(abs(sol_prec-sol_ip),N,N))
% % title("difference PResQPASS-IP")
% print('solutionContact','-dpdf','-painters');
% 
% 
% 
% width = 17; height = 5;
% figure(Units="centimeters", Position=[4 4 width height], PaperUnits="centimeters", PaperSize=[width height]);
% % subplot(1,3,1); 
% % semilogy(abs(obj-obj_prec(end))); hold all; semilogy(abs(obj_prec-obj_prec(end))); title("objective")
% % ylabel("$\|Ax_k-b\|_2^2$")
% 
% subplot(1,2,1); 
% semilogy(res/res(1), markings{1}, Color=colors{1}, MarkerIndices=1:10:200); 
% hold on;
% semilogy(res_prec/res_prec(1), markings{4}, Color=colors{4}, MarkerIndices=1:10:200); 
% title("Residual")
% ylabel("${\|r_k\|}/{\|r_0\|}$",Rotation=0,HorizontalAlignment="right")
% xlabel("Iteration $k$")
% legend("No preconditioning", "Preconditioning", Location="southwest")
% 
% subplot(1,2,2)
% % plot(iters, markings{1}, Color=colors{1}, MarkerIndices=1:10:200); 
% % hold on; 
% plot(iters_prec, markings{4}, Color=colors{4}, MarkerIndices=1:10:200); 
% title("Iterations")
% ylabel("\# QPAS iterations")
% xlabel("Iteration $k$")


%% Objective animation
%Uncomment the code-block below to animate the evolution of the solution
%per iteration

% figure(Units="centimeters",Position=[0 5 30 15])
% for i=1:size(X_prec,2)
% mesh(X,Y,reshape(X_prec(:,i),N,N))
% zlim([0,0.1])
% title(i)
% pause(0.1)
% drawnow;
% end

%% Slides UA
% FWETBlue = "#006CA9";
% FWETBlueLight = "#97C0DF";
% UABlue = "#002E65";
% 
% [X,Y] = meshgrid(x,x);
% 
% width = 10; height = 11;
% markings = {'-o', '-square','-diamond','-^','-*','-pentagram'};
% colors = {'#1B9E77','#D95F02','#7570B3','#E7298A','#66A61E','#E6AB02'};
% 
% figure(Units="centimeters", Position=[4 4 width height], PaperUnits="centimeters", PaperSize=[width height]);
% 
% % for i=1:2
% k=1;
% for i=[1:39, 40:10:size(X_prec,2)]
% subplot(2,1,1)
% mesh(X,Y,reshape(X_prec(:,i),N,N))
% zlim([0,0.1])
% 
% set(gca, 'Box', 'on')
% ax=gca;
% ax.YColor = UABlue;
% ax.XColor = UABlue;
% ax.ZColor = UABlue;
% ax.GridColor = UABlue;
% ax.MinorGridColor = FWETBlueLight;
% set(gca,'FontName', 'SansSerif')
% title("Preconditioning", Color=FWETBlue)
% 
% subplot(2,1,2)
% mesh(X,Y,reshape(X_unprec(:,i),N,N))
% zlim([0,0.1])
% 
% set(gca, 'Box', 'on')
% ax=gca;
% ax.YColor = UABlue;
% ax.XColor = UABlue;
% ax.ZColor = UABlue;
% ax.GridColor = UABlue;
% ax.MinorGridColor = FWETBlueLight;
% set(gca,'FontName', 'SansSerif')
% title("No preconditioning", Color=FWETBlue)
% 
% sgtitle(strcat("k = ",num2str(i)), Color=FWETBlue)
% % pause(0.1)
% % drawnow;
% print(sprintf('animate/balloon%d.pdf',k),'-dpdf','-painters');
% k=k+1;
% end