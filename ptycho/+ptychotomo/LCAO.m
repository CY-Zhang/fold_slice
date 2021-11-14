% Calculate ZincBlende Bandstructure from Walter Harrison's book with an
% added extra s* state for the Conduction band (labelled as t since * inside
% variable names is not allowed) as in Vogl's paper

%Onsite energies are given for cation then anion (same for elemental)
% e.g. ecs is energy for cation s,  eap is energy for anion p.

hm = 7.62; % hbar^2/m in eV.A^2

% Cubic lattice constant
alat = 5.65; % GaAs lattice constant

% GaAs
% Onsite Matrix elements (in eV)
eas = -8.3431;
eap = 1.0414;
eat = 8.5914;

ecs = -2.6569;
ecp = 3.6686;
ect = 6.7386;


%vectors to neighboring cations
d1 = alat/4 * [1 1 1];
d2 = alat/4 * [1 -1 -1];
d3 = alat/4 * [-1 1 -1];
d4 = alat/4 * [-1 -1 1];
d = [d1; d2; d3; d4];

%phase factors
g0 = inline('exp(i*k*transpose(d(1,:)))+exp(i*k*transpose(d(2,:)))+exp(i*k*transpose(d(3,:)))+exp(i*k*transpose(d(4,:)))','k','d');
g1 = inline('exp(i*k*transpose(d(1,:)))+exp(i*k*transpose(d(2,:)))-exp(i*k*transpose(d(3,:)))-exp(i*k*transpose(d(4,:)))','k','d');
g2 = inline('exp(i*k*transpose(d(1,:)))-exp(i*k*transpose(d(2,:)))+exp(i*k*transpose(d(3,:)))-exp(i*k*transpose(d(4,:)))','k','d');
g3 = inline('exp(i*k*transpose(d(1,:)))-exp(i*k*transpose(d(2,:)))-exp(i*k*transpose(d(3,:)))+exp(i*k*transpose(d(4,:)))','k','d');

%Interatomic Matrix Elements (eV) using Harrison's universal parameters
dn = norm(d1);

% Composite Matrix elements
Ess = -1.4054* hm/dn^2;
Esp = 1.0392*(hm/dn^2);
Exx = 0.3111*(hm/dn^2);
Exy = 0.8298*(hm/dn^2);
Etp = 0.9549* hm/dn^2;  % Harrison parameter for s*ps from Vtps*dn^2/hm
Ett  = 0;  % In Dow's model no coupling between s* states on adjacent atoms is allowed


% Zone boundary and number of k points
qbz = 2*pi/alat;
nk=20;
j=1;

E=1;clear E;
kx=1; clear kx;
for q=0:qbz/nk:qbz,
k = [q 0 0];

%LCAO hamiltonian for the zincblende structure
% Calculate Upper Half of the LCAO Hamiltonian Matrix and use its
% Hermiticity to get lower half
Hu = [[ ecs/2   Ess *g0(k,d)  0             0                   0                   Esp*g1(k,d)   Esp * g2(k,d)     Esp* g3(k,d) 0  0];
      [ 0        eas/2     -Esp * conj(g1(k,d))  -Esp * conj(g2(k,d))  -Esp * conj(g3(k,d))  0           0               0       0  0];
      [ 0       0       ecp/2               0                   0                   Exx * g0(k,d) Exy * g3(k,d)  Exy * g2(k,d)   0  -Etp * g1(k,d)]; 
      [ 0       0       0                  ecp/2                0                   Exy * g3(k,d) Exx * g0(k,d)  Exy * g1(k,d)   0  -Etp * g2(k,d) ];
      [ 0       0       0                  0                   ecp/2                Exy * g2(k,d) Exy * g1(k,d)  Exx * g0(k,d)   0  -Etp * g3(k,d)];
      [ 0       0       0                  0                   0              eap/2         0               0         Etp * g1(k,d)  0 ];
      [ 0       0       0                  0                   0               0           eap/2             0        Etp * g2(k,d)  0  ];
      [ 0       0       0                  0                   0               0           0               eap/2      Etp * g3(k,d)  0  ];
      [ 0       0       0                  0                   0               0           0                 0      eat/2        Ett *g0(k,d) ];
      [ 0       0       0                  0                   0               0           0                0       0           eat/2];
      ];

%Its adjoint
Hd = (Hu)';

%The full Hermitian Matrix
H = Hu+Hd;

%Calculate Eigenvalues
E(j,:) = eig(H)';
kx(j)=q;
j=j+1;

end;


plot(kx,E,'k');
ylabel('Energy (eV)');

