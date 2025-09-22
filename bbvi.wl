(* ::Package:: *)

BeginPackage["bbvi`"]
adam::usage="adam";
gd::usage="gd";
gd0::usage="gd0";
gd1::usage="gd1";
m2M::usage="Kq";
Uq::usage="Uq";
Uqq::usage="Uqq";
outbnd::usage="check for out of bound"
SAMPLES::usage="SAMPLES";
ITERATIONS::usage="ITERATIONS";
STEPSIZE::usage="STEPSIZE";
PARTICLES::usage="PARTICLES";
STEPS::usage="STEPS";
(*INTERVAL::usage="INTERVAL";*)


D1::usage="D1[f,x]";

Begin["`Private`"]
SAMPLES=5;
PARTICLES=3;
STEPS=3;
(*INTERVAL=1001;
*)
outbnd[q_]:=False;
ITERATIONS=2^10;
STEPSIZE=0.001;


D1[f_,x_List?VectorQ]:=D[f,{x,1}];
k2ij[k_]:=Module[{i,j,k1},k1=k-1;i=Floor[(-1+Sqrt[1+8k1])/2];j=k1-(i(i+1))/2;{i+1,j+1}];
m2M[m_]:=Module[{n,M,k,i,j},n=1/2 (-1+Sqrt[1+8 Length[m]]);M=ConstantArray[0,{n,n}];For[k=1,k<=Length[m],k++,{i,j}=k2ij[k];M[[i,j]]=m[[k]];M[[j,i]]=m[[k]]];M];
M2m[M_]:=Module[{m,k,i,j},m=ConstantArray[0,Length[M](Length[M]+1)/2];For[k=1,k<=Length[m],k++,{i,j}=k2ij[k];m[[k]]=If[i==j,M[[i,j]],M[[i,j]]+M[[j,i]]]];m]

(*\:71b5*)
y[m_]:=Module[{Q,\[Sigma],M},
M=m2M[m];
Q=M . Transpose[M];
\[Sigma]=Eigenvalues[Q];
Total[1/2 Log[2\[Pi] E \[Sigma]]]]

Dy[m_]:=Module[{Q,\[Lambda],v,n,M1,s,t,j,k,M},
M=m2M[m];

Q=M . Transpose[M];
\[Lambda]=Eigenvalues[Q];
v=Eigenvectors[Q];
n=Length[M];
M1=ConstantArray[0,{n,n}];
For[s=1,s<=n,s++,
For[t=1,t<=n,t++,
For[j=1,j<=n,j++,
For[k=1,k<=n,k++,
If[Abs[\[Lambda][[j]]]>10^-9,
M1[[s,t]]=M1[[s,t]]+1/ \[Lambda][[j]] (v[[j,s]]v[[j,k]]M[[k,t]])]];
]]];M2m[M1]]

(*\:6839\:636e\:6b63\:6001\:5206\:5e03\:6837\:672cz\:751f\:6210\:4e00\:6b21-logP(x)\:ff0c\:8fd4\:56def(\[Mu],m),Subscript[f, \[Mu]](\[Mu],m),Subscript[f, m](\[Mu],m)*)
f[U_,Uq_,\[Mu]_,m_,z_]:=Module[{x,M,uq},
M=m2M[m];
x=M . z+\[Mu];
uq=Apply[Uq,x];
{Apply[U,x],uq,M2m[Outer[Times,uq,z]]}]

grad[U_,Uq_,params_,Dim_]:=Module[{\[Mu],m,z,loss,r,\[Mu]d,md},
\[Mu]=params[[;;Dim]];
m=params[[Dim+1;;]];
\[Mu]d=ConstantArray[0,Dim];
md=ConstantArray[0,Length[m]];
loss=0;
For[i=1,i<=SAMPLES,i++,
z=RandomVariate[NormalDistribution[],Dim];
r=f[U,Uq,\[Mu],m,z];
\[Mu]d=\[Mu]d+r[[2]];
md=md+r[[3]];
loss=loss+r[[1]]];
\[Mu]d=\[Mu]d/SAMPLES;
md=md/SAMPLES-Dy[m];
loss=loss/SAMPLES-y[m];
{loss,Join[\[Mu]d,md]}]

(*no bbvi,i.e. no sigma*)
gd0[U_,Uq_,Dim_,x0_]:=Module[{INTERVAL,loss,g,numiters=ITERATIONS,Dim1,tuned=False,stepsize=STEPSIZE,x,xs={},i,j,k,UE,ES,s,S,decaydt=0.1},
INTERVAL=Round[numiters/100];
Dim1=Dim+Dim (Dim+1)/2;
xs=Table[{},PARTICLES];
x=If[x0=={},RandomVariate[UniformDistribution[],{PARTICLES,Dim}],x0];
For[i=1,i<=numiters,i++,

ES={};
For[j=1,j<=PARTICLES,j++,
UE={};
For[k=1,k<=STEPS,k++,
g=Apply[Uq,x[[j]]];
loss=Apply[U,x[[j]]];
x[[j]]=x[[j]]-stepsize g;
AppendTo[UE,loss];
AppendTo[xs[[j]],x[[j]]];
];
AppendTo[ES,UE]];
s=Union[Flatten[Table[Ordering[ES[[i]],1],{i,1,PARTICLES}]]];
S=Union[Flatten[Table[Ordering[ES[[i]],-1],{i,1,PARTICLES}]]];
If[s=={1}&&S=={STEPS},stepsize=stepsize/(1+decaydt)];
If[s=={STEPS}&&S=={1},stepsize=stepsize(1+decaydt)];
If[Mod[i,INTERVAL]==0,Print[i,s,S,ES,stepsize]]];
xs]

(*no multiparticle, no tuning*)
gd1[U_,Uq_,Dim_,x0_]:=Module[{INTERVAL,loss,g,numiters=ITERATIONS,tuned=False,stepsize=STEPSIZE,x,i,xs},
INTERVAL=Round[numiters/10];
xs={};
x=If[x0=={},RandomVariate[UniformDistribution[],Dim+Dim (Dim+1)/2],x0];
For[i=1,i<=numiters,i++,
{loss,g}=grad[U,Uq,x,Dim];
x=x-stepsize g;
AppendTo[xs,x];
If[Mod[i,INTERVAL]==0,Print[i]]];
xs]

adam[U_,Uq_,Dim_,x0_]:=Module[{numiters=ITERATIONS,stepsize=STEPSIZE,fs,g,b1=0.9,b2=0.999,eps=10^-8,x,m,v,xs=List[],i},
x=If[x0=={},RandomVariate[UniformDistribution[],Dim+Dim (Dim+1)/2],x0];
m=Table[0,Length[x]];
v=Table[0,Length[x]];
For[i=1,i<=numiters,i++,
If[Mod[i,INTERVAL]==0,Print[i]];
{fs,g}=grad[U,Uq,x,Dim];
m=(1-b1) g+b1 m;
v=(1-b2) g^2+b2 v;
mhat=m/(1-b1^(i+1));
vhat=v/(1-b2^(i+1));
x=x-stepsize mhat/(Sqrt[vhat]+eps);
xs=Append[xs,x]];
(*
*)
If[s=={1}&&S=={STEPS},dt=dt/(1+decaydt),dt=dt(1+decaydt)];

xs]


End[]


EndPackage[]
