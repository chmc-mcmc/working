(* ::Package:: *)

BeginPackage["sampler2`"]
(*\:6b63\:4ea4*)
hmc::usage = "hmc[U,Uq,Uqq,Dim,BURNIN,EPISODE,qinit]";

Kp::usage="Kp";
Kq::usage="Kq";
K::usage="K[p,q,i]";
Uq::usage="Uq";
Uqq::usage="Uqq";


D1::usage="D1[f,x]";
D2::usage="D2[f,x]";
STEPS::usage="simulation steps"
INTERVAL::usage="INTERVAL for print"
outbnd::usage="check for out of bound"
RATIODT::usage="updating ratio";
RATIOENERGY::usage="updating ratio";
LOWLEVEL::usage="lowest allowable average acceptance probability";
HIGHLEVEL::usage="highest allowable average acceptance probability";

Begin["`Private`"]
STEPS=3;
INTERVAL=1001;
outbnd[q_]:=False;
RATIODT=.1;
RATIOENERGY=.1;
LOWLEVEL=0.1;
HIGHLEVEL=0.9;


D1[f_,x_List?VectorQ]:=D[f,{x,1}];
D2[f_,x_List?VectorQ]:=D[f,{x,2}];

K[p_,q_,idx_,r_,Uqq_]:=Module[{W},
W=getW[q,idx,r,Uqq];
1/2 p . (W . p)];
Kp[p_,q_,idx_,r_,Uqq_]:=Module[{W},
W=getW[q,idx,r,Uqq];
W . p];


getW[q_,idx_,r_,Uqq_]:=Module[{ve,e,s,eig,\[Lambda]},
eig=Eigensystem[Apply[Uqq,q]];
ve=eig[[2]];
e=eig[[1]];
\[Lambda]=Table[If[j!=idx||e[[j]]==0,0,Sign[e[[j]]] Abs[e[[j]]]^(-r)],{j,1,Length[e]}];
(*\[Lambda]=Sign[e] Abs[e]^(-r);*)
Transpose[ve] . DiagonalMatrix[\[Lambda]] . ve];


hmc[U_,Uq_,Uqq_,Dim_,BURNIN_,ITERATIONS_,PARTICLES_,r_:0.5,qinit_:{}]:=Module[{dts,Htotals,level,dt,qAll,dq1,pAll,UE,ES,Utotal,Ktotal,hi,lo,Htotal,step,dp,dq,s,S,AS,anybad,KtotalNew,p,q0,p0,\[Alpha],q,j,bad,i,q1,decayenergy=RATIOENERGY,decaydt=RATIODT, QS=ConstantArray[0,{PARTICLES( ITERATIONS-BURNIN),Dim}]},
If[qinit!={},
qAll=qinit,
qAll=RandomVariate[UniformDistribution[],{PARTICLES,Dim}]];
Utotal=Sum[Apply[ U,qAll[[i]]],{i,1,PARTICLES}];
(*smallest scale*)

dts=Min/@Transpose[Table[Abs[Eigenvalues[Apply[Uqq,qAll[[i]]]]]^(-.5),{i,1,PARTICLES}]];
Htotals=Table[If[Utotal==0,1,2Utotal],Dim];
level=1;
For[j=1,j\[LessSlantEqual]ITERATIONS,j++,
pAll=RandomVariate[NormalDistribution[0,1],{PARTICLES,Dim}];
KtotalNew = Sum[Apply[K,{pAll[[i]],qAll[[i]],level,r,Uqq}],{i,1,PARTICLES}];
Utotal=Sum[Apply[ U,qAll[[i]]],{i,1,PARTICLES}];
Htotal=Htotals[[level]];
dt=dts[[level]];
Ktotal=Htotal-Utotal;
pAll=pAll Sqrt[Abs[Ktotal/KtotalNew]];
AS={};ES={};
anybad=False;
For[i=1,i<=PARTICLES,i++,
bad=False;
p0=pAll[[i]];
q0=qAll[[i]];
q=q0;
p=p0;
(*simulation*)
(*burnin: multi step*)
If[j<=BURNIN,
UE={Apply[U,q]};
For[step=1,step<=STEPS && Not[bad],step++,
dq=Apply[Kp,{p,q,level,r,Uqq}];
dp=-Apply[Uq,q];
q=q+dt dq ;
p=p+dt dp;
If[outbnd[q],bad=True];
UE=Append[UE,Apply[U,q]]];
ES=Append[ES,UE];
anybad=anybad||bad];


(*one step*)
q=q0;
p=p0;
dq=Apply[Kp,{p,q,level,r,Uqq}];
If[True,
q1=q+Sqrt[STEPS] dt dq;
dq1=Apply[Kp,{p,q1,level,r,Uqq}];
q=q+Sqrt[STEPS] dt .5(dq+dq1),
q=q+Sqrt[STEPS] dt dq];
If[outbnd[q],bad=True;q=q0];
(*Metropolis*)
\[Alpha]=If[bad,0,Exp[Clip[Apply[U,q0]-Apply[U,q],{-30,0}]]];
If[\[Alpha]<RandomVariate[UniformDistribution[]],q=q0];
qAll[[i]]=q;
AS=Append[AS,N[\[Alpha]]];
If[j>BURNIN,QS[[(j-BURNIN-1)PARTICLES+i,;;]]=q]];

If[j<=BURNIN && Not[anybad],

s=Union[Flatten[Table[Ordering[ES[[i]],1],{i,1,PARTICLES}]]];
S=Union[Flatten[Table[Ordering[ES[[i]],-1],{i,1,PARTICLES}]]];
If[s=={1,STEPS+1}&&S=={1,STEPS+1},dt=dt(1+decaydt)];
If[s=={1}&&S=={STEPS+1},dt=dt/(1+decaydt)];

hi=Mean[AS]>HIGHLEVEL;
lo=Mean[AS]<LOWLEVEL;
If[hi,Htotal=(Htotal-Utotal)(1+decayenergy)+Utotal];
If[lo,Htotal=(Htotal-Utotal)/(1+decayenergy)+Utotal];
dts[[level]]=dt;
Htotals[[level]]=Htotal;
];
If[Mod[j,INTERVAL]==0,Print[Row[{j,Utotal,Ktotal,Htotal,dt,Htotals,dts,Mean[AS],StandardDeviation[AS],s,S,anybad,level},"     "]]];
level=Mod[level,Dim]+1];
QS];


End[]


EndPackage[]



