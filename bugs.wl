(* ::Package:: *)

BeginPackage["bugs`"]
dunif::usage = "dunif";
dbern::usage = "dbern";
dbeta::usage="dbeta";
dbin::usage="dbin";
dlnorm::usage="dlnorm";
dnorm::usage="dnorm";
dpois::usage="dpois";
dexp::usage="dexp";
dcat::usage="dcat";
dweib::usage="dweib";
rweib::usage="rweib";
dunif::usage="dunif";
dt::usage="dt";
dgamma::usage="dgamma";
dmulti::usage="dmulti";
ddirch::usage="ddirch";
logit::usage="logit";
sigmoid::usage="sigmoid";
rbin::usage="rbin";
dpar::usage = "dpar";


Begin["`Private`"]


dunif[x_,a_,b_]:=If[x>=a&&x<=b ,-Log[1/(b-a)],-Infinity]
ddirch[x_,\[Theta]_]:=-(Total[(\[Theta]-1)Log[x]]+LogGamma[Total[\[Theta]]]-Total[LogGamma[\[Theta]]])
dmulti[x_,\[Theta]_]:=-(Total[x Log[\[Theta]]]+Log[Factorial[Total[x]]]-Total[Log[Factorial[x]]])
dbern[x_,p_]:=(-1+x) Log[1-p]-x Log[p]
dbeta[x_,a_,b_]:=Simplify[PowerExpand[-Log[((1-x)^(-1+b) x^(-1+a))/Beta[a,b]]]]
dbin[x_,\[Theta]_,n_]:=(-n+x) Log[1-\[Theta]]-x Log[\[Theta]]-Log[Binomial[n,x]]
dlnorm[x_,\[Mu]_,\[Tau]_]:=PowerExpand[-Log[Simplify[PDF[LogNormalDistribution[\[Mu],1/Sqrt[\[Tau]]],x],x>0]]]
dnorm[x_,mu_,tau_]:=1/2 tau (-mu+x)^2+1/2 (Log[2]+Log[\[Pi]])-Log[tau]/2
dexp[x_,\[Theta]_]=x \[Theta]-Log[\[Theta]];
dpois[x_,\[Theta]_]:=\[Theta]-x Log[\[Theta]]+Log[x!]
dcat[x_,\[Theta]_]:=-Log[\[Theta][[x]]]
dweib[x_,a_,b_]:=b x^a-Log[a]-Log[b]+Log[x]-a Log[x]
rweib[x_,a_,b_]:=b x^a
dunif[x_,l_,r_]:=1
logit[\[Theta]_]=Log[\[Theta]/(1-\[Theta])];
sigmoid[x_]=1/(1+E^-x);
dt[x_,\[Mu]_,\[Tau]_,k_]=1/2 (Log[k]-Log[\[Tau]]-(1+k) (Log[k]-Log[k+(x-\[Mu])^2 \[Tau]])+2 Log[Beta[k/2,1/2]]);
dgamma[x_,a_,b_]=b x-a Log[b]+Log[x]-a Log[x]+Log[Gamma[a]];
rbin[p_,n_]:=RandomVariate[BinomialDistribution[n,p]]
dpar[x_,a_,b_]=-Log[a]-a Log[b]-(-1-a) Log[x]


End[]


EndPackage[]



