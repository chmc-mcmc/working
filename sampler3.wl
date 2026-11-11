(* ::Package:: *)

BeginPackage["sampler3`"]

(* ---------- \:5bfc\:51fa\:63a5\:53e3\:7684 usage ---------- *)
hmc::usage        = "hmc[U, Uq, Uqq, Dim, BURNIN, ITERATIONS, qinit]";
Kp::usage         = "Kp[p, q, k]";
K::usage          = "K[p, q, k]";
U::usage          = "U[q]";
Uq::usage         = "Uq[q]";
Uqq::usage        = "Uqq[q]";
D2::usage         = "D2[f, x]";
D1::usage         = "D1[f, x]";
psrf::usage       = "psrf[QS]";
PARTICLES::usage  = "MCMC chains / particles";
dt0::usage        = "Time step for HMC";
STEPS::usage      = "Leapfrog steps";
INTERVAL::usage   = "Print interval";
outbnd::usage     = "outbnd[q]";
DECAYDT::usage    = "dt decay ratio";
DECAYENERGY::usage = "Energy decay ratio";
LOWLEVEL::usage   = "Lower acceptance threshold";
HIGHLEVEL::usage  = "Upper acceptance threshold";

Begin["`Private`"]

(* ---------- \:9ed8\:8ba4\:8d85\:53c2\:6570 ---------- *)
PARTICLES   = 3;
STEPS       = 5;
dt0         = 10^-1;
INTERVAL    = 1001;
outbnd[q_]  := False;
DECAYDT     = .1;
DECAYENERGY = .1;
LOWLEVEL    = 0.1;
HIGHLEVEL   = 0.9;

(* ---------- \:4e3b\:91c7\:6837\:5668 ---------- *)
hmc[U_, Uq_, Uqq_, Dim_, BURNIN_, ITERATIONS_, qinit_] := Module[{
qAll, pAll, UE, ES, Utotal, Ktotal, dq, hi, lo,
Htotal, Htotal1,Htotal2, s, S, AS, dt1, dt2, k, KtotalNew,
dt = dt0, p, q0, p0, alpha, q, j, bad, i,
decayenergy = DECAYENERGY, decaydt = DECAYDT, ii = 1,
QS = ConstantArray[0, {PARTICLES (ITERATIONS - BURNIN), Dim}]
},

If[qinit =!= {},
qAll = qinit,
qAll = RandomVariate[NormalDistribution[0, 1], {PARTICLES, Dim}]
];

Utotal = Sum[Apply[U, qAll[[i]]], {i, 1, PARTICLES}];
Htotal = If[Utotal == 0, 1, 2 Utotal];
Htotal1 = Htotal;
dt1 = dt;
Htotal2 = Htotal;
dt2 = dt;
k = 1;

For[j = 1, j <= ITERATIONS, j++,
pAll = RandomVariate[NormalDistribution[0, 1], {PARTICLES, Dim}];

KtotalNew = Sum[Apply[K, {pAll[[i]], qAll[[i]], k}], {i, 1, PARTICLES}];
Utotal    = Sum[Apply[U, qAll[[i]]], {i, 1, PARTICLES}];

If[k == 1,
Htotal = Htotal1; dt = dt1,
Htotal = Htotal2; dt = dt2
];

Ktotal = Htotal - Utotal;
pAll   = pAll Sqrt[Abs[Ktotal / KtotalNew]];

AS = {}; ES = {};

For[i = 1, i <= PARTICLES, i++,
bad = False;
p   = pAll[[i]];
q   = qAll[[i]];
q0  = q;
p0  = p;

If[j <= BURNIN, UE = {Apply[U, q]}];

Do[
q  = q + dt Apply[Kp, {p, q, k}];
p  = p - dt Apply[Uq, q];

If[outbnd[q],q = q0];
If[j <= BURNIN, UE = Append[UE, Apply[U, q]]],
STEPS];

If[j <= BURNIN, ES = Append[ES, UE]];

(* \:5355\:6b21\:5927\:6b65\:957f\:63d0\:6848 *)
q  = q0;
p  = p0;
dq = Apply[Kp, {p, q, k}];
q  = q + Sqrt[STEPS] dt dq;

If[outbnd[q], bad = True; q = q0];

alpha = If[bad, 0, Exp[Clip[Apply[U, q0] - Apply[U, q], {-10, 0}]]];
If[alpha < RandomVariate[UniformDistribution[]], q = q0];
qAll[[i]] = q;

AS = Append[AS, N[alpha]];

If[j > BURNIN,
QS[[ii, ;;]] = q;
ii = ii + 1
]
];

(* burn-in \:9636\:6bb5\:6839\:636e\:80fd\:91cf\:6781\:5c0f/\:6781\:5927\:4f4d\:7f6e\:8c03\:6574 dt *)
If[j <= BURNIN,
s = Union[Flatten[Table[Ordering[ES[[i]], 1],  {i, 1, PARTICLES}]]];
S = Union[Flatten[Table[Ordering[ES[[i]], -1], {i, 1, PARTICLES}]]];
If[s == {1, STEPS + 1} && S == {1, STEPS + 1}, dt = dt (1 + decaydt)];
If[s == {1}          && S == {STEPS + 1},    dt = dt/(1 + decaydt)];
If[k == 1, dt1 = dt, dt2 = dt]
];

(* \:6839\:636e\:63a5\:53d7\:7387\:8c03\:6574\:80fd\:91cf\:5e38\:6570 Htotal *)
If[j <= BURNIN,
hi = Quantile[AS, .5] > HIGHLEVEL;
lo = Quantile[AS, .5] < LOWLEVEL;

If[hi || lo,
Utotal = Sum[Apply[U, qAll[[i]]], {i, 1, PARTICLES}]];

If[hi,
Htotal = (Htotal - Utotal) (1 + decayenergy) + Utotal,
If[lo,
Htotal = (Htotal - Utotal)/(1 + decayenergy) + Utotal]];

If[k == 1, Htotal1 = Htotal, Htotal2 = Htotal]
];

If[Mod[j, INTERVAL] == 0,
Print[Row[{j, k, Ktotal, KtotalNew, Mean[AS], StandardDeviation[AS],
Utotal, s, S, dt, Htotal, dt, hi, lo}, "     "]]
];

If[k == 1, k = 2, k = 1]

];

QS
];

(* ---------- \:52a8\:80fd\:4e0e\:8d28\:91cf\:77e9\:9635 ---------- *)
K[p_, q_, k_] := Module[{w},
w = getW[q, Uqq, k];
1/2 p . Dot[w . p]
];

Kp[p_, q_, k_] := Module[{w},
w = getW[q, Uqq, k];
w . p
];

getW[q_, Uqq_, k_] := Module[{w = Apply[Uqq, q], ve, e, lambda},
ve = Eigenvectors[w];
e  = Eigenvalues[w];
lambda  = Table[If[e[[j]] == 0, 0,
If[k == 1, 1, Sign[e[[j]]]] / Sqrt[Abs[e[[j]]]]],
{j, 1, Length[e]}];
Transpose[ve] . DiagonalMatrix[lambda] . ve
];

(* ---------- \:6570\:503c\:5bfc\:6570\:5de5\:5177 ---------- *)
D2[f_, x_List?VectorQ] := N[D[f, {x, 2}]];
D1[f_, x_List?VectorQ] := N[D[f, {x, 1}]];

(* ---------- PSRF (Gelman[Dash]Rubin \:6536\:655b\:8bca\:65ad) ---------- *)
psrf[QS_] := Module[{QQ, theta, sigma2, B, W, V, Rs,
len = Length[QS], chains = PARTICLES},
QQ = Table[QS[[i ;; Length[QS] ;; PARTICLES, ;;]], {i, 1, PARTICLES}];
theta  = Table[Mean[QQ[[j]]], {j, 1, chains}];
sigma2 = Table[Variance[QQ[[j]]], {j, 1, chains}];
B  = (Total[Table[theta[[j]] - Mean[theta], {j, 1, chains}]^2] len/chains)/(chains - 1);
W  = Mean[sigma2];
V  = (len/chains - 1)/(len/chains) W + (chains + 1)/(len/chains chains) B;
Rs = Sqrt[V/W];
Rs
];

End[]

EndPackage[]
