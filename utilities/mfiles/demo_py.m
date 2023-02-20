%Number of grid points on [0,1]^2 
%i.e. uniform mesh with step h=1/(N-1)
clear;
rng('default');
rng(1);
N = 2^4;

%Create mesh (only needed for plotting)
[X,Y] = meshgrid(0:(1/(N-1)):1);

%Parameters of covariance C = (-Laplacian + tau^2 I)^(-alpha)
%Note that we need alpha > d/2 (here d= 2) 
%Laplacian has zero Neumann boundry
%alpha and tau control smoothness; the bigger they are, the smoother the
%function
alpha = 2;
tau = 3;

%Generate random coefficients from N(0,C)
norm_a = gaussrnd(alpha, tau, N);

%Exponentiate it, so that a(x) > 0
%Now a ~ Lognormal(0, C)
%This is done so that the PDE is elliptic

%Solve PDE: - div(a(x)*grad(p(x))) = f(x)
lognorm_p = solve_gwf(norm_a);

%Plot coefficients and solutions
subplot(1,2,1)
surf(X,Y, norm_a); 
view(2); 
shading interp;
colorbar;
subplot(1,2,2)
surf(X,Y,lognorm_p); 
view(2); 
shading interp;
colorbar;
