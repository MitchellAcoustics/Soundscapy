CircE.BFGS <-
function(R,
                        v.names,
                        
                        m,
                        N,r=1,
                        equal.com=FALSE,
                        equal.ang=FALSE, 
                        mcsc="unconstrained",     
                        start.values="IFA",
                        ci.level=0.95,
                        factr=1e9,pgtol=0,lmm=NULL,
                  iterlim=250, upper=NULL,lower=NULL,print.level=1,file=NULL,title="Circumplex Estimation",try.refit.BFGS=FALSE){
      if(!is.null(file))  sink(file,append=FALSE,split=TRUE)        
  

if(is.null(N)) stop('No default value for argument N. Specify sample size.')
if(is.null(m)) stop('No default value for argument m. Specify the number of free parameters in the Fourier correlation function (m>=1).')

if(m<=0) stop('The number of free parameters in the Fourier correlation function must be m>=1')

if (!is.null(lower)&equal.ang==TRUE) stop('You are trying to impose bounds on equally spaced polar angles! Set equal.ang=FALSE')
if (!is.null(upper)&equal.ang==TRUE) stop('You are trying to impose bounds on equally spaced polar angles! Set equal.ang=FALSE')
if(!is.null(upper) & !is.null(lower)){
for (i in 1:length(upper)){
if(upper[i]<lower[i]) stop ('lower bound greater than corresponding upper bound.')
} 
}


      p=dim(R)[1]
      is.triang <- function(R) {
      is.matrix(R) && (nrow(R) == ncol(R)) && 
            (all(0 == R[upper.tri(R)])) || (all(0 == R[lower.tri(R)]))
        }    
      is.symm <- function(R) {
        is.matrix(R) && (nrow(R) == ncol(R)) && all(R == t(R))
        }
    if (is.triang(R)) R <- R + t(R) - diag(diag(R))
    if (!is.symm(R)) stop('R MUST BE A SQUARE TRIANGULAR OR SYMMETRIC MATRIX !')
          
      k=3
      K=pi/180
cat("Date:",date(),"\n")
cat("Data:",title,"\n")
cat("Model:")
if(equal.com==TRUE & equal.ang==TRUE) {cat("Constrained model: equal spacing and equal radius \n")}
if(equal.com==TRUE & equal.ang==FALSE) {cat("Equal radius \n")}
if(equal.com==FALSE & equal.ang==TRUE) {cat("Equal spacing \n")}
if(equal.com==FALSE & equal.ang==FALSE){cat("Unconstrained model \n")}
cat("Reference variable at 0 degree:",v.names[r],"\n")
cat("\n")

ifa<-function(rr,mm) {
	if (length(which(eigen(rr)$values < 0)) != 0) {
            cat("WARNING!", "\n")
            cat("INPUT COVARIANCE/CORRELATION MATRIX IS NOT POSITIVE DEFINITE.", "\n")
            cat("STARTING VALUES CANNOT BE COMPUTED USING 'IFA': SET start.values='PFA'", "\n")
            stop("Make sure the listwise, not pairwise, missing data treatment has been selected in computing the input matrix\n")
             }

rinv <- solve(rr) 
sm2i <- diag(rinv)
smrt <- sqrt(sm2i)
dsmrt <- diag(smrt)
rsr <- dsmrt %*% rr %*% dsmrt
reig <- eigen(rsr)
vlamd <- reig$va
vlamdm <- vlamd[1:mm]
qqm <- as.matrix(reig$ve[, 1:mm])
theta <- mean(vlamd[(mm + 1):nrow(qqm)])
dg <- sqrt(vlamdm - theta)
fac <- diag(1/smrt) %*% qqm %*% diag(dg)
list(vlamd = vlamd, theta = theta,
fac = fac)
  }

pfa<-function (rr, mm =3, min.err = 0.001, max.iter = iterlim, 
    symmetric = TRUE, warnings = TRUE) 
{

        if (!is.matrix(rr)) {
            rr <- as.matrix(rr)
        }
        sds <- sqrt(diag(rr))
        rr <- rr/(sds %o% sds)
    
    
    
    rr.mat <- rr

    orig <- diag(rr)
    comm <- sum(diag(rr.mat))
    err <- comm
    i <- 1
    comm.list <- list()
    while (err > min.err) {
        eigens <- eigen(rr.mat, symmetric = symmetric)
        if (mm > 1) {
            loadings <- eigens$vectors[, 1:mm] %*% diag(sqrt(eigens$values[1:mm]))
        }
        else {
            loadings <- eigens$vectors[, 1] * sqrt(eigens$values[1])
        }
        model <- loadings %*% t(loadings)
        new <- diag(model)
        comm1 <- sum(new)
        diag(rr.mat) <- new
        err <- abs(comm - comm1)
        if (is.na(err)) {
            warning("imaginary eigen value condition encountered in factor.pa, exiting")
            break
        }
        comm <- comm1
        comm.list[[i]] <- comm1
        i <- i + 1
        if (i > max.iter) {
            if (warnings) {
                message("maximum iteration exceeded")
            }
            err <- 0
        }
    }
    if (!is.double(loadings)) {
        warning("the matrix has produced imaginary results -- proceed with caution")
        loadings <- matrix(as.double(loadings), ncol = mm)
    }
    if (mm > 1) {
        maxabs <- apply(apply(loadings, 2, abs), 2, which.max)
        sign.max <- vector(mode = "numeric", length = mm)
        for (i in 1:mm) {
            sign.max[i] <- sign(loadings[maxabs[i], i])
        }
        loadings <- loadings %*% diag(sign.max)
    }
    else {
        mini <- min(loadings)
        maxi <- max(loadings)
        if (abs(mini) > maxi) {
            loadings <- -loadings
        }
        loadings <- as.matrix(loadings)
    }
    loadings[loadings == 0] <- 10^-15
    model <- loadings %*% t(loadings)


list(fac=loadings)
}

        
start.valuesA=function(R,k,start.value=start.values){
        
        one=matrix(1,p,1)
        Lambda=if(start.value=="IFA")ifa(R,k)$fac else if(start.value=="PFA")pfa(rr=R, mm =k, min.err = 0.001, max.iter = iterlim)$fac

        Diagzeta=diag(diag(Lambda%*%t(Lambda)))
        Dzeta=sqrt(Diagzeta)
        
        uniq=diag(1,p,p)-(Lambda%*%t(Lambda))
        Dpsi=diag(diag(uniq))
        Dv=solve(Dzeta)%*%Dpsi
        
        Lambdax=solve(Dzeta)%*%Lambda
        Lambdaxhat=1/p*(t(Lambdax)%*%one)
        C=1/p*t(Lambdax-one%*%t(Lambdaxhat))%*%(Lambdax-one%*%t(Lambdaxhat))
        U=eigen(C)$vectors
        Lambdatilde=Lambdax%*%U
        
        if(equal.ang==FALSE){
                
        L..=Lambdatilde[,1:2]
        L..[,]=0
        for(i in 1:p){
                
                L..[i,]=Lambdatilde[i,1:2]/sqrt(Lambdatilde[i,1]^2+Lambdatilde[i,2]^2)
                
                }
   
   r=r
   
   sin.angles=rep(0,p)
   for(i in 1:p){
        
        sin.angles[i]=L..[i,2]*L..[r,1]-L..[i,1]*L..[r,2]
        
        }

   
   polar.angles=rep(0,p)
   for(i in 1:p){
        
        if(sin.angles[i]>=0){
        polar.angles[i]=L..[i,1]*L..[r,1]+L..[i,2]*L..[r,2]
        polar.angles[i]=acos(polar.angles[i])
        }
        else if(sin.angles[i]<0){
        polar.angles[i]=L..[i,1]*L..[r,1]+L..[i,2]*L..[r,2]
        polar.angles[i]=2*pi-acos(polar.angles[i])
        }
        }
  polar.angles=ifelse(polar.angles=="NaN",0,polar.angles) 
  polar.angles=polar.angles/K}

     if(equal.ang==TRUE){
           polar.angles=rep(0,p)
   for(i in 1:p){
        polar.angles[i]=(i-1)*(2*pi)/p
        }
        
polar.angles=polar.angles/K}
        

if(mcsc=="unconstrained"){
betas=matrix(0,1,m+1);
betas[,1]=(1/p*sum(Lambdatilde[,3]))^2
betas[,2]=1-betas[,1]
if(m>k){betas[,3:m]=0} ;if(m==k){ betas[,3]=0}
betas=betas/betas[,2]
                         }


if(mcsc=="-1"){
betas=matrix(0,1,m+1);
betas[,c(seq(1,(m+1),by=2))]=0
betas[,2]=1-betas[,1]
if(m>k){betas[,3:m]=0} ;if(m==k){ betas[,3]=0}
betas=betas/betas[,2] }



if(mcsc=="0"){
betas=matrix(0,1,m+1);
betas[,c(seq(1,(m+1)/2,by=2))]=1/(length(c(seq(1,(m+1)/2,by=2)))*2)
betas[,2]=1-betas[,1]
if(m>k){betas[,3:m]=0} ;if(m==k){ betas[,3]=0}
betas=betas/betas[,2] }


z=sqrt(diag(Lambda%*%t(Lambda)))
uniq=diag(1,p,p)-(diag(z^2))
        Dpsi=diag(diag(uniq))
        Dv=solve(Dzeta)%*%Dpsi
v=diag(Dv)
if(equal.com==TRUE){v=mean(v)} else v=v
if(equal.ang==FALSE)par=c(polar.angles[-c(1)]*K,c(betas[-c(2)]),v,z)
 if(equal.ang==TRUE) par=c(c(betas[-c(2)]),v,z)
 attributes(par)=list(parA=par,polar.angles=polar.angles,betas=betas)
 }
 
 parA=start.valuesA(R,k)$parA
 betas=start.valuesA(R,k)$betas
 
 
 ASK<-
function (arg) 
{
         value <- readline(paste("Continue? (YES/NO)", deparse(substitute(arg)), 
            ": "))
        if (value == "NO") 
           stop("STOP CURRENT COMPUTATION.",call.=FALSE)
     
}
if(length(parA)>1/2*(p*(p+1))){
	cat("ERROR: NUMBER OF PARAMETERS,",length(parA),", GREATER THAN NUMBER OF NONDUPLICATED ELEMENTS OF INPUT MATRIX,",1/2*(p*(p+1)),"\n* THE MODEL IS UNDERIDENTIFIED: Consider simplifying the model by reducing 'm' or by using equality constraints ('equal.com=TRUE' and/or 'equal.ang=TRUE'); \n* THE HESSIAN MATRIX MAY NOT BE POSITIVE DEFINITE:  THE STANDARD ERRORS OF THE MODEL PARAMETER ESTIMATES MAY NOT BE CALCULATED; \n* THE PROGRAM  MAY GENERATE NON-SENSICAL ESTIMATES OR DISPLAY VERY LARGE STANDARD ERRORS; \n* DEGREE OF FREEDOM < 0: SEVERAL FIT INDEXES CANNOT BE CALCULATED;...")
	ASK()}
if(length(parA)==1/2*(p*(p+1))){
	cat("ERROR: NUMBER OF PARAMETERS,",length(parA),", EQUAL TO THE NUMBER OF NONDUPLICATED ELEMENTS OF INPUT MATRIX,",1/2*(p*(p+1)),"\n* THE MODEL IS JUST IDENTIFIED (SATURATED): Consider simplifying the model by reducing 'm' or by using equality constraints ('equal.com=TRUE' and/or 'equal.ang=TRUE'); \n* THE STANDARD ERRORS OF THE MODEL PARAMETER ESTIMATES MAY NOT BE TRUSTWORTHY; \n* DEGREE OF FREEDOM = 0: SEVERAL FIT INDEXES CANNOT BE CALCULATED;...")
	ASK()}
 
 
append<-
function (x, values, after = length(x)) 
{
    lengx <- length(x)
    if (after <= 0) 
        c(values, x)
    else if (after >= lengx) 
        c(x, values)
    else c(x[1:after], values, x[(after + 1):lengx])
}


objective1max<- 
function(par){
        ang1=par[1:(p-1)]
        ang=append(ang1,0,r-1)  
      if(mcsc=="unconstrained"){if(m==1){b=  c(par[((p-1)+1)],1)} else b=  c(par[((p-1)+1)],1,par[((p-1)+2):((p-1)+(m))]); b=b/sum(b)}
      if(mcsc=="-1" ){if(m==1){b=  c(0,1)} else if(m>=3){b=  c(par[((p-1)+1)],1,par[((p-1)+2):((p-1)+(m))]);b[c(seq(1,(m+1),by=2))]=0; b=b/sum(b)} else {b=  c(par[((p-1)+1)],1,par[((p-1)+2):((p-1)+(m))]);b[c(seq(1,(m+1),by=2))]=0; b=b/sum(b)}}
      if(mcsc=="0" ){ if(m==1){b=  c(0.5,0.5)} else if(m>=3){b=  c(par[((p-1)+1)],1,par[((p-1)+2):((p-1)+(m))]);b[1]=sum(b[seq(2,m+1,by=2)])-sum(b[seq(3,m+1,by=2)]); b=b/sum(b)} else {b=  c(par[((p-1)+1)],1,par[((p-1)+2):((p-1)+(m))]);b[1]=sum(b[seq(2,m+1,by=2)])-b[3]; b=b/sum(b)}}


        
        v=  par[((p-1)+(m)+1)]
                                  z=  par[((p-1)+(m)+1+1):((p-1)+(m)+1+p)]
  
 K=pi/180
 M=matrix(c(0),p,p,byrow=TRUE)
  for(i in 1:p){
        for(j in 1:p){
        M[i,j]=c(b[-c(1)])%*%cos(c(1:m)*(ang[j]-ang[i]))
        }}
 Pc=M+matrix(b[1],p,p)
 Dv=diag(v,p)
 Dz=diag(z); f=-1*(sum(diag(R%*%solve(Dz%*%(Pc+Dv)%*%Dz)))+log(det(Dz%*%(Pc+Dv)%*%Dz))-log(det(R))-p);
 S1=Dz%*%(Pc+Dv)%*%Dz;S2=diag(1/sqrt(diag(S1)))%*%Dz;S=S2%*%(Pc+Dv)%*%S2
 attributes(f)=list(f=f,S=S,Cs=Dz%*%(Pc+Dv)%*%Dz,Pc=Pc)
 f}
 
objective1gr<-
function(par){
        ang1=par[1:(p-1)]
        ang=append(ang1,0,r-1)  
      if(mcsc=="unconstrained"){if(m==1){alpha=  c(par[((p-1)+1)],1)} else alpha=  c(par[((p-1)+1)],1,par[((p-1)+2):((p-1)+(m))]); b=alpha/sum(alpha)}
      if(mcsc=="-1" ){if(m==1){b=alpha=  c(0,1)} else if(m>=3){alpha=  c(par[((p-1)+1)],1,par[((p-1)+2):((p-1)+(m))]);alpha[c(seq(1,(m+1),by=2))]=0; b=alpha/sum(alpha)} else {alpha=  c(par[((p-1)+1)],1,par[((p-1)+2):((p-1)+(m))]);alpha[c(seq(1,(m+1),by=2))]=0; b=alpha/sum(alpha)}}
      if(mcsc=="0" ){ if(m==1){b=alpha=  c(0.5,0.5)} else if(m>=3){alpha=  c(par[((p-1)+1)],1,par[((p-1)+2):((p-1)+(m))]);alpha[1]=sum(alpha[seq(2,m+1,by=2)])-sum(alpha[seq(3,m+1,by=2)]); b=alpha/sum(alpha)} else {alpha=  c(par[((p-1)+1)],1,par[((p-1)+2):((p-1)+(m))]);alpha[1]=sum(alpha[seq(2,m+1,by=2)])-alpha[3]; b=alpha/sum(alpha)}}          
        v=  par[((p-1)+(m)+1)]
                                  z=  par[((p-1)+(m)+1+1):((p-1)+(m)+1+p)]
 K=pi/180
 M=matrix(c(0),p,p,byrow=TRUE)
  for(i in 1:p){
        for(j in 1:p){
        M[i,j]=c(b[-c(1)])%*%cos(c(1:m)*(ang[j]-ang[i]))
        }}
 Pc=M+matrix(b[1],p,p)
 Dv=diag(v,p)
 Dz=diag(z); 
 S=Dz%*%(Pc+Dv)%*%Dz
 
 hessian=matrix(0,length(par),length(par))

 gradientz=rep(0,p);Dpz=matrix(0,p*p,p)
for(i in 1:p){
        J=matrix(0,p,p)
        J[i,i]=1;
        dp=J%*%(Pc+Dv)%*%Dz+Dz%*%(Pc+Dv)%*%J
        gradientz[i]=1*sum(diag(  solve(S)%*%(R-S)%*%solve(S)%*%dp  ))
        Dpz[,i]=c(dp)
        }
gradientv=rep(0,p);Dpv=matrix(0,p*p,1)
        J=diag(1,p,p) 
        dp=J%*%(Dz)^2
        gradientv=1*sum(diag(  solve(S)%*%(R-S)%*%solve(S)%*%dp  ))
        Dpv=c(dp)
       
 if((mcsc=="unconstrained")){       
gradientalph=rep(0,(m+1));Dpalph=matrix(0,p*p,length(alpha))
for(i in 2:(m+1)){
        M=matrix(c(0),p,p,byrow=TRUE)
  for(z in 1:p){
        for(j in 1:p){
        M[z,j]=(    (1/sum(alpha)-alpha[i]/sum(alpha)^2)*cos((i-1)*(ang[j]-ang[z]))  -1*(alpha[1]/sum(alpha)^2+(alpha[-c(1)]/sum(alpha))^2%*%cos(c(1:m)*(ang[j]-ang[z])))       )*Dz[z,z]*Dz[j,j] 
        }}
        dp=M
        gradientalph[i]=1*sum(diag(  solve(S)%*%(R-S)%*%solve(S)%*%dp  ))
        Dpalph[,i]=c(dp)
        }
        
        
        for(i in 1:1){
        M=matrix(c(0),p,p,byrow=TRUE)
  for(z in 1:p){
        for(j in 1:p){
        M[z,j]=(  1/sum(alpha)-alpha[i]/sum(alpha)^2 -1*sum( (alpha[-c(1)]/sum(alpha)^2)*cos(c(1:m)*(ang[j]-ang[z]) ) ) )*Dz[z,z]*Dz[j,j]
        }}
dp=M
        gradientalph[i]=1*sum(diag(  solve(S)%*%(R-S)%*%solve(S)%*%dp  ))
        
        Dpalph[,i]=c(dp)
        }  
}

 if((mcsc=="-1")){      
gradientalph=rep(0,(m+1));Dpalph=matrix(0,p*p,length(alpha))
if(  m>=3 ){ 
for(i in seq(4,m+1,by=2)){
        M=matrix(c(0),p,p,byrow=TRUE)
  for(z in 1:p){
        for(j in 1:p){
        M[z,j]=(   (1/sum(alpha)-alpha[i]/sum(alpha)^2)*cos((i-1)*(ang[j]-ang[z]))  -1*(alpha[1]/sum(alpha)^2+(alpha[-c(1)]/sum(alpha))^2%*%cos(c(1:m)*(ang[j]-ang[z])))      )*Dz[z,z]*Dz[j,j]  
        }}
        dp=M
        gradientalph[i]=1*sum(diag(  solve(S)%*%(R-S)%*%solve(S)%*%dp  ))
        Dpalph[,i]=c(dp)
        }
           }
   
        
                 }
if((mcsc=="-1" & m<=2)){ 
gradientalph=rep(0,(m+1));Dpalph=matrix(0,p*p,length(alpha))
}
if((mcsc=="0")){ 
gradientalph=rep(0,(m+1));Dpalph=matrix(0,p*p,length(alpha))
if(  m>=3 ){ 
for(i in seq(4,m+1,by=2)){
        M=matrix(c(0),p,p,byrow=TRUE)
  for(z in 1:p){
        for(j in 1:p){
        M[z,j]=(  1/sum(alpha) 
        
        - (sum(alpha[seq(2,m+1,by=2)])-sum(alpha[seq(3,m+1,by=2)]))*1/sum(alpha) 
        
        -(sum(alpha[-c(1,i)]*1/sum(alpha)*cos(c(seq(1,m,by=1)[-c(i-1)])*(ang[j]-ang[z]))))  
        
        +(1/sum(alpha)-alpha[i]*1/sum(alpha))*cos(c(i-1)*(ang[j]-ang[z]))   )*Dz[z,z]*Dz[j,j]
        }}
        dp=M
        gradientalph[i]=1*sum(diag(  solve(S)%*%(R-S)%*%solve(S)%*%dp  ))
        Dpalph[,i]=c(dp)
        } 
}
if(  m>=2 ){ 
for(i in seq(3,m+1,by=2)){
        M=matrix(c(0),p,p,byrow=TRUE)
  for(z in 1:p){
        for(j in 1:p){
        M[z,j]=(  1/sum(alpha)*cos(c(i-1)*(ang[j]-ang[z])) -(1/sum(alpha))  )*Dz[z,z]*Dz[j,j]
        }}
        dp=M
        gradientalph[i]=1*sum(diag(  solve(S)%*%(R-S)%*%solve(S)%*%dp  ))
        Dpalph[,i]=c(dp)
        }
}
  

if((mcsc=="0" & m==1)){ 
gradientalph=rep(0,(m+1));Dpalph=matrix(0,p*p,length(alpha))
} 
}

gradientang=rep(0,p);Dpang=matrix(0,p*p,length(ang))
for(i in 1:p){
        M=matrix(c(0),p,p,byrow=TRUE)
  for(z in 1:p){
        for(j in 1:p){
                
                if(z==i) s=1
                if(z!=i) s=0
                if(j==i) sj=1
                if(j!=i) sj=0
        M[z,j]= (s*c(1:m)%*%(b[-c(1)]*sin(c(1:m)*(ang[j]-ang[i]))) - sj*c(1:m)%*%(b[-c(1)]*sin(c(1:m)*(ang[i]-ang[z]))))*Dz[z,z]*Dz[j,j]
        }}
        dp=M;
        Dpang[,i]=c(dp)
        
        gradientang[i]=1*sum(diag(  solve(S)%*%(R-S)%*%solve(S)%*%dp  ))
        
        
        };Dpang=Dpang[,-c(1)];
        gradient=c(gradientang[-c(r)], if((mcsc=="unconstrained") | (mcsc=="-1" & m>=3) | (mcsc=="0" & m>=2)){       
gradientalph[-c(2)]} else if((mcsc=="-1" & m==1) | (mcsc=="0" & m==1)){0} else if((mcsc=="-1" & m==2)){c(0,0)},gradientv,gradientz)
        

 
 gradient}

objective1hess<-
function(par){
        ang1=par[1:(p-1)]
        ang=append(ang1,0,r-1)  
      if(mcsc=="unconstrained"){if(m==1){alpha=  c(par[((p-1)+1)],1)} else alpha=  c(par[((p-1)+1)],1,par[((p-1)+2):((p-1)+(m))]); b=alpha/sum(alpha)}
      if(mcsc=="-1" ){if(m==1){b=alpha=  c(0,1)} else if(m>=3){alpha=  c(par[((p-1)+1)],1,par[((p-1)+2):((p-1)+(m))]);alpha[c(seq(1,(m+1),by=2))]=0; b=alpha/sum(alpha)} else {alpha=  c(par[((p-1)+1)],1,par[((p-1)+2):((p-1)+(m))]);alpha[c(seq(1,(m+1),by=2))]=0; b=alpha/sum(alpha)}}
      if(mcsc=="0" ){if(m==1){b=alpha=c(0.5,0.5)} else if(m>=3){alpha=  c(par[((p-1)+1)],1,par[((p-1)+2):((p-1)+(m))]);alpha[1]=sum(alpha[seq(2,m+1,by=2)])-sum(alpha[seq(3,m+1,by=2)]); b=alpha/sum(alpha)} else {alpha=  c(par[((p-1)+1)],1,par[((p-1)+2):((p-1)+(m))]);alpha[1]=sum(alpha[seq(2,m+1,by=2)])-alpha[3]; b=alpha/sum(alpha)}}

        v=  par[((p-1)+(m)+1)]
                                  z=  par[((p-1)+(m)+1+1):((p-1)+(m)+1+p)]
         K=pi/180
 M=matrix(c(0),p,p,byrow=TRUE)
  for(i in 1:p){
        for(j in 1:p){
        M[i,j]=c(b[-c(1)])%*%cos(c(1:m)*(ang[j]-ang[i]))
        }}
 Pc=M+matrix(b[1],p,p)
 Dv=diag(v,p)
 Dz=diag(z); 

 S=Dz%*%(Pc+Dv)%*%Dz;S2=diag(1/sqrt(diag(S)))%*%Dz;S1=S2%*%(Pc+Dv)%*%S2
 hessian=matrix(0,length(par),length(par))

 gradientz=rep(0,p);Dpz=matrix(0,p*p,p)
for(i in 1:p){
        J=matrix(0,p,p)
        J[i,i]=1;
        dp=J%*%(Pc+Dv)%*%Dz+Dz%*%(Pc+Dv)%*%J
        gradientz[i]=1*sum(diag(  solve(S)%*%(R-S)%*%solve(S)%*%dp  ))
        Dpz[,i]=c(dp)
        }
gradientv=rep(0,p);Dpv=matrix(0,p*p,1)
        J=diag(1,p,p) 
        dp=J%*%(Dz)^2
        gradientv=1*sum(diag(  solve(S)%*%(R-S)%*%solve(S)%*%dp  ))
        Dpv=c(dp)


 if((mcsc=="unconstrained")){       
gradientalph=rep(0,(m+1));Dpalph=matrix(0,p*p,length(alpha))
for(i in 2:(m+1)){
        M=matrix(c(0),p,p,byrow=TRUE)
  for(z in 1:p){
        for(j in 1:p){
        M[z,j]=(   (1/sum(alpha)-alpha[i]/sum(alpha)^2)*cos((i-1)*(ang[j]-ang[z]))      )*Dz[z,z]*Dz[j,j]
        }}
        dp=M
        gradientalph[i]=1*sum(diag(  solve(S)%*%(R-S)%*%solve(S)%*%dp  ))
        Dpalph[,i]=c(dp)
        }
        
        
        for(i in 1:1){
        M=matrix(c(0),p,p,byrow=TRUE)
  for(z in 1:p){
        for(j in 1:p){
        M[z,j]=(  1/sum(alpha)-alpha[i]/sum(alpha)^2 -1*sum( (alpha[-c(1)]/sum(alpha)^2)*cos(c(1:m)*(ang[j]-ang[z]) ) ) )*Dz[z,z]*Dz[j,j]
        }}
dp=M
        gradientalph[i]=1*sum(diag(  solve(S)%*%(R-S)%*%solve(S)%*%dp  ))
        
        Dpalph[,i]=c(dp)
        };if(mcsc=="unconstrained"){Dpalph=Dpalph[,-c(2)]} ;  
}
 if((mcsc=="-1")){      
gradientalph=rep(0,(m+1));Dpalph=matrix(0,p*p,length(alpha))
if(  m>=3 ){ 
for(i in seq(4,m+1,by=2)){
        M=matrix(c(0),p,p,byrow=TRUE)
  for(z in 1:p){
        for(j in 1:p){
        M[z,j]=(   (1/sum(alpha)-alpha[i]/sum(alpha)^2)*cos((i-1)*(ang[j]-ang[z]))  -1*(alpha[1]/sum(alpha)^2+(alpha[-c(1)]/sum(alpha))^2%*%cos(c(1:m)*(ang[j]-ang[z])))     )*Dz[z,z]*Dz[j,j]  
        }}
        dp=M
        gradientalph[i]=1*sum(diag(  solve(S)%*%(R-S)%*%solve(S)%*%dp  ))
        Dpalph[,i]=c(dp)
        }
           Dpalph=Dpalph[,-c(2)]}
   
    
if(m<=2){ 
gradientalph=rep(0,(m+1));Dpalph=matrix(0,p*p,length(alpha))
Dpalph=Dpalph[,-c(2)]   }
                 
                  }


if((mcsc=="0")){ 
gradientalph=rep(0,(m+1));Dpalph=matrix(0,p*p,length(alpha))
if(  m>=3 ){ 
for(i in seq(4,m+1,by=2)){
        M=matrix(c(0),p,p,byrow=TRUE)
  for(z in 1:p){
        for(j in 1:p){
        M[z,j]=(  1/sum(alpha) 
        
        - (sum(alpha[seq(2,m+1,by=2)])-sum(alpha[seq(3,m+1,by=2)]))*1/sum(alpha) 
        
        -(sum(alpha[-c(1,i)]*1/sum(alpha)*cos(c(seq(1,m,by=1)[-c(i-1)])*(ang[j]-ang[z]))))  
        
        +(1/sum(alpha)-alpha[i]*1/sum(alpha))*cos(c(i-1)*(ang[j]-ang[z]))   )*Dz[z,z]*Dz[j,j]
        }}
        dp=M
        gradientalph[i]=1*sum(diag(  solve(S)%*%(R-S)%*%solve(S)%*%dp  ))
        Dpalph[,i]=c(dp)
        } 
}
if(  m>=2 ){
for(i in seq(3,m+1,by=2)){
        M=matrix(c(0),p,p,byrow=TRUE)
  for(z in 1:p){
        for(j in 1:p){
        M[z,j]=(  1/sum(alpha)*cos(c(i-1)*(ang[j]-ang[z])) -(1/sum(alpha))  )*Dz[z,z]*Dz[j,j]
        }}
        dp=M
        gradientalph[i]=1*sum(diag(  solve(S)%*%(R-S)%*%solve(S)%*%dp  ))
        Dpalph[,i]=c(dp)
        }
}
  Dpalph=Dpalph[,-c(2)]

}
if( (mcsc=="0" & m==1)){ 
gradientalph=rep(0,(m+1));Dpalph=matrix(0,p*p,length(b))
 Dpalph=Dpalph[,-c(2)]
}


gradientang=rep(0,p);Dpang=matrix(0,p*p,length(ang))
for(i in 1:p){
        M=matrix(c(0),p,p,byrow=TRUE)
  for(z in 1:p){
        for(j in 1:p){
                
                if(z==i) s=1
                if(z!=i) s=0
                if(j==i) sj=1
                if(j!=i) sj=0
        M[z,j]= (s*c(1:m)%*%(b[-c(1)]*sin(c(1:m)*(ang[j]-ang[i]))) - sj*c(1:m)%*%(b[-c(1)]*sin(c(1:m)*(ang[i]-ang[z]))))*Dz[z,z]*Dz[j,j]
        }}
        dp=M;
        Dpang[,i]=c(dp)
        
        gradientang[i]=1*sum(diag(  solve(S)%*%(R-S)%*%solve(S)%*%dp  ))
        
        
        };Dpang=Dpang[,-c(1)];
        gradient=c(gradientang[-c(r)], if((mcsc=="unconstrained") | (mcsc=="-1" & m>=3) | (mcsc=="0" & m>=2)){       
gradientalph[-c(2)]}  else if((mcsc=="-1" & m==1) | (mcsc=="0" & m==1)){0} else if((mcsc=="-1" & m==2)){c(0,0)},gradientv,gradientz)
        if((mcsc=="unconstrained") | (mcsc=="-1" & m>=1) | (mcsc=="0" & m>=1)){
        DerivAnal=data.frame(Dpang,Dpalph,Dpv,Dpz);
        for(i in 1:length(par)){
        for(j in 1:length(par)){
     
                      hessian[i,j]=-1*sum(diag(  solve(S)%*%matrix(DerivAnal[,i],p,p)%*%solve(S)%*%matrix(DerivAnal[,j],p,p) ))

  }}} 
 
 
  hessian}




objective2max<-
function(par){
        ang1=par[1:(p-1)]
        ang=append(ang1,0,r-1)  
      if(mcsc=="unconstrained"){if(m==1){b=  c(par[((p-1)+1)],1)} else b=  c(par[((p-1)+1)],1,par[((p-1)+2):((p-1)+(m))]);b=b/sum(b)}
      if(mcsc=="-1" ){if(m==1){b=  c(0,1)} else if(m>=3){b=  c(par[((p-1)+1)],1,par[((p-1)+2):((p-1)+(m))]);b[c(seq(1,(m+1),by=2))]=0; b=b/sum(b)} else {b=  c(par[((p-1)+1)],1,par[((p-1)+2):((p-1)+(m))]);b[c(seq(1,(m+1),by=2))]=0; b=b/sum(b)}}
      if(mcsc=="0" ){ if(m==1){b=  c(0.5,0.5)} else if(m>=3){b=  c(par[((p-1)+1)],1,par[((p-1)+2):((p-1)+(m))]);b[1]=sum(b[seq(2,m+1,by=2)])-sum(b[seq(3,m+1,by=2)]); b=b/sum(b)} else {b=  c(par[((p-1)+1)],1,par[((p-1)+2):((p-1)+(m))]);b[1]=sum(b[seq(2,m+1,by=2)])-b[3]; b=b/sum(b)}}

       
        v=  par[((p-1)+(m)+1):((p-1)+(m)+p)]
                                  z=  par[((p-1)+(m)+p+1):((p-1)+(m)+p+p)]
 K=pi/180
 M=matrix(c(0),p,p,byrow=TRUE)
  for(i in 1:p){
        for(j in 1:p){
        M[i,j]=c(b[-c(1)])%*%cos(c(1:m)*(ang[j]-ang[i]))
        }}
 Pc=M+matrix(b[1],p,p)
 Dv=diag(v,p)
 Dz=diag(z); f=-1*(sum(diag(R%*%solve(Dz%*%(Pc+Dv)%*%Dz)))+log(det(Dz%*%(Pc+Dv)%*%Dz))-log(det(R))-p);
 S1=Dz%*%(Pc+Dv)%*%Dz;S2=diag(1/sqrt(diag(S1)))%*%Dz;S=S2%*%(Pc+Dv)%*%S2
 attributes(f)=list(f=f,S=S,Cs=Dz%*%(Pc+Dv)%*%Dz,Pc=Pc)
 f}
 
objective2gr<-
function(par){
        ang1=par[1:(p-1)]
        ang=append(ang1,0,r-1)  
        if(mcsc=="unconstrained"){if(m==1){alpha=  c(par[((p-1)+1)],1)} else alpha=  c(par[((p-1)+1)],1,par[((p-1)+2):((p-1)+(m))]);b=alpha/sum(alpha)}
      if(mcsc=="-1" ){if(m==1){b=alpha=  c(0,1)} else if(m>=3){alpha=  c(par[((p-1)+1)],1,par[((p-1)+2):((p-1)+(m))]);alpha[c(seq(1,(m+1),by=2))]=0; b=alpha/sum(alpha)} else {alpha=  c(par[((p-1)+1)],1,par[((p-1)+2):((p-1)+(m))]);alpha[c(seq(1,(m+1),by=2))]=0; b=alpha/sum(alpha)}}
      if(mcsc=="0" ){ if(m==1){b=alpha= c(0.5,0.5)} else if(m>=3){alpha=  c(par[((p-1)+1)],1,par[((p-1)+2):((p-1)+(m))]);alpha[1]=sum(alpha[seq(2,m+1,by=2)])-sum(alpha[seq(3,m+1,by=2)]); b=alpha/sum(alpha)} else {alpha=  c(par[((p-1)+1)],1,par[((p-1)+2):((p-1)+(m))]);alpha[1]=sum(alpha[seq(2,m+1,by=2)])-alpha[3]; b=alpha/sum(alpha)}}          

        v=  par[((p-1)+(m)+1):((p-1)+(m)+p)]
                                  z=  par[((p-1)+(m)+p+1):((p-1)+(m)+p+p)]
 K=pi/180
 M=matrix(c(0),p,p,byrow=TRUE)
  for(i in 1:p){
        for(j in 1:p){
        M[i,j]=c(b[-c(1)])%*%cos(c(1:m)*(ang[j]-ang[i]))
        }}
 Pc=M+matrix(b[1],p,p)
 Dv=diag(v,p)
 Dz=diag(z); 
 S=Dz%*%(Pc+Dv)%*%Dz
 
 hessian=matrix(0,length(par),length(par))

 gradientz=rep(0,p);Dpz=matrix(0,p*p,p)
for(i in 1:p){
        J=matrix(0,p,p)
        J[i,i]=1;
        dp=J%*%(Pc+Dv)%*%Dz+Dz%*%(Pc+Dv)%*%J
        gradientz[i]=1*sum(diag(  solve(S)%*%(R-S)%*%solve(S)%*%dp  ))
        Dpz[,i]=c(dp)
        }
gradientv=rep(0,p);Dpv=matrix(0,p*p,p)
for(i in 1:p){
        J=matrix(0,p,p)
        J[i,i]=1; 
        dp=J%*%(Dz)^2
        gradientv[i]=1*sum(diag(  solve(S)%*%(R-S)%*%solve(S)%*%dp  ))
        Dpv[,i]=c(dp)
        }

 if((mcsc=="unconstrained")){       
gradientalph=rep(0,(m+1));Dpalph=matrix(0,p*p,length(alpha))
for(i in 2:(m+1)){
        M=matrix(c(0),p,p,byrow=TRUE)
  for(z in 1:p){
        for(j in 1:p){
        M[z,j]=(   (1/sum(alpha)-alpha[i]/sum(alpha)^2)*cos((i-1)*(ang[j]-ang[z]))  -1*(alpha[1]/sum(alpha)^2+(alpha[-c(1)]/sum(alpha))^2%*%cos(c(1:m)*(ang[j]-ang[z])))     )*Dz[z,z]*Dz[j,j]
        }}
        dp=M
        gradientalph[i]=1*sum(diag(  solve(S)%*%(R-S)%*%solve(S)%*%dp  ))
        Dpalph[,i]=c(dp)
        }
        
        
        for(i in 1:1){
        M=matrix(c(0),p,p,byrow=TRUE)
  for(z in 1:p){
        for(j in 1:p){
        M[z,j]=(  1/sum(alpha)-alpha[i]/sum(alpha)^2 -1*sum( (alpha[-c(1)]/sum(alpha)^2)*cos(c(1:m)*(ang[j]-ang[z]) ) ) )*Dz[z,z]*Dz[j,j]
        }}
dp=M
        gradientalph[i]=1*sum(diag(  solve(S)%*%(R-S)%*%solve(S)%*%dp  ))
        
        Dpalph[,i]=c(dp)
        }
}

 if((mcsc=="-1")){      
gradientalph=rep(0,(m+1));Dpalph=matrix(0,p*p,length(alpha))
if(  m>=3 ){ 
for(i in seq(4,m+1,by=2)){
        M=matrix(c(0),p,p,byrow=TRUE)
  for(z in 1:p){
        for(j in 1:p){
        M[z,j]=(    (1/sum(alpha)-alpha[i]/sum(alpha)^2)*cos((i-1)*(ang[j]-ang[z]))  -1*(alpha[1]/sum(alpha)^2+(alpha[-c(1)]/sum(alpha))^2%*%cos(c(1:m)*(ang[j]-ang[z])))    )*Dz[z,z]*Dz[j,j]
        }}
        dp=M
        gradientalph[i]=1*sum(diag(  solve(S)%*%(R-S)%*%solve(S)%*%dp  ))
        Dpalph[,i]=c(dp)
        }
           }
   
        
                 }
   
        
                 
if((mcsc=="-1" & m<=2)){ 
gradientalph=rep(0,(m+1));Dpalph=matrix(0,p*p,length(alpha))
}
if((mcsc=="0")){ 
gradientalph=rep(0,(m+1));Dpalph=matrix(0,p*p,length(alpha))
if(  m>=3 ){ 
for(i in seq(4,m+1,by=2)){
        M=matrix(c(0),p,p,byrow=TRUE)
  for(z in 1:p){
        for(j in 1:p){
        M[z,j]=(  1/sum(alpha) 
        
        - (sum(alpha[seq(2,m+1,by=2)])-sum(alpha[seq(3,m+1,by=2)]))*1/sum(alpha) 
        
        -(sum(alpha[-c(1,i)]*1/sum(alpha)*cos(c(seq(1,m,by=1)[-c(i-1)])*(ang[j]-ang[z]))))  
        
        +(1/sum(alpha)-alpha[i]*1/sum(alpha))*cos(c(i-1)*(ang[j]-ang[z]))   )*Dz[z,z]*Dz[j,j]
        }}
        dp=M
        gradientalph[i]=1*sum(diag(  solve(S)%*%(R-S)%*%solve(S)%*%dp  ))
        Dpalph[,i]=c(dp)
        } 
}
if(  m>=2 ){ 
for(i in seq(3,m+1,by=2)){
        M=matrix(c(0),p,p,byrow=TRUE)
  for(z in 1:p){
        for(j in 1:p){
        M[z,j]=(  1/sum(alpha)*cos(c(i-1)*(ang[j]-ang[z])) -(1/sum(alpha))  )*Dz[z,z]*Dz[j,j]
        }}
        dp=M
        gradientalph[i]=1*sum(diag(  solve(S)%*%(R-S)%*%solve(S)%*%dp  ))
        Dpalph[,i]=c(dp)
        }
}
 
}
if((mcsc=="0" & m==1)){ 
gradientalph=rep(0,(m+1));Dpalph=matrix(0,p*p,length(alpha))
 
}
gradientang=rep(0,p);Dpang=matrix(0,p*p,length(ang))
for(i in 1:p){
        M=matrix(c(0),p,p,byrow=TRUE)
  for(z in 1:p){
        for(j in 1:p){
                
                if(z==i) s=1
                if(z!=i) s=0
                if(j==i) sj=1
                if(j!=i) sj=0
        M[z,j]= (s*c(1:m)%*%(b[-c(1)]*sin(c(1:m)*(ang[j]-ang[i]))) - sj*c(1:m)%*%(b[-c(1)]*sin(c(1:m)*(ang[i]-ang[z]))))*Dz[z,z]*Dz[j,j]
        }}
        dp=M;
        Dpang[,i]=c(dp)
        
        gradientang[i]=1*sum(diag(  solve(S)%*%(R-S)%*%solve(S)%*%dp  ))
        
        
        };Dpang=Dpang[,-c(1)];
        gradient=c(gradientang[-c(r)], if((mcsc=="unconstrained") | (mcsc=="-1" & m>=3) | (mcsc=="0" & m>=2)){       
gradientalph[-c(2)]} else if((mcsc=="-1" & m==1) | (mcsc=="0" & m==1)){0} else if((mcsc=="-1" & m==2)){c(0,0)},gradientv,gradientz)
        
 
 
 gradient}

objective2hess<-
function(par){
        ang1=par[1:(p-1)]
        ang=append(ang1,0,r-1)  
        if(mcsc=="unconstrained"){if(m==1){alpha=  c(par[((p-1)+1)],1)} else alpha=  c(par[((p-1)+1)],1,par[((p-1)+2):((p-1)+(m))]);b=alpha/sum(alpha)}
     if(mcsc=="-1" ){if(m==1){b=alpha=  c(0,1)} else if(m>=3){alpha=  c(par[((p-1)+1)],1,par[((p-1)+2):((p-1)+(m))]);alpha[c(seq(1,(m+1),by=2))]=0; b=alpha/sum(alpha)} else {alpha=  c(par[((p-1)+1)],1,par[((p-1)+2):((p-1)+(m))]);alpha[c(seq(1,(m+1),by=2))]=0; b=alpha/sum(alpha)}}
      if(mcsc=="0" ){if(m==1){b=alpha=  c(0.5,0.5)} else if(m>=3){alpha=  c(par[((p-1)+1)],1,par[((p-1)+2):((p-1)+(m))]);alpha[1]=sum(alpha[seq(2,m+1,by=2)])-sum(alpha[seq(3,m+1,by=2)]); b=alpha/sum(alpha)} else {alpha=  c(par[((p-1)+1)],1,par[((p-1)+2):((p-1)+(m))]);alpha[1]=sum(alpha[seq(2,m+1,by=2)])-alpha[3]; b=alpha/sum(alpha)}}

        v=  par[((p-1)+(m)+1):((p-1)+(m)+p)]
                                  z=  par[((p-1)+(m)+p+1):((p-1)+(m)+p+p)]
 K=pi/180
 M=matrix(c(0),p,p,byrow=TRUE)
  for(i in 1:p){
        for(j in 1:p){
        M[i,j]=c(b[-c(1)])%*%cos(c(1:m)*(ang[j]-ang[i]))
        }}
 Pc=M+matrix(b[1],p,p)
 Dv=diag(v,p)
 Dz=diag(z); 

 S=Dz%*%(Pc+Dv)%*%Dz;S2=diag(1/sqrt(diag(S)))%*%Dz;S1=S2%*%(Pc+Dv)%*%S2
 hessian=matrix(0,length(par),length(par))

 gradientz=rep(0,p);Dpz=matrix(0,p*p,p)
for(i in 1:p){
        J=matrix(0,p,p)
        J[i,i]=1;
        dp=J%*%(Pc+Dv)%*%Dz+Dz%*%(Pc+Dv)%*%J
        gradientz[i]=1*sum(diag(  solve(S)%*%(R-S)%*%solve(S)%*%dp  ))
        Dpz[,i]=c(dp)
        }
gradientv=rep(0,p);Dpv=matrix(0,p*p,p)
for(i in 1:p){
        J=matrix(0,p,p)
        J[i,i]=1; 
        dp=J%*%(Dz)^2
        gradientv[i]=1*sum(diag(  solve(S)%*%(R-S)%*%solve(S)%*%dp  ))
        Dpv[,i]=c(dp)
        }


 if((mcsc=="unconstrained")){       
gradientalph=rep(0,(m+1));Dpalph=matrix(0,p*p,length(alpha))
for(i in 2:(m+1)){
        M=matrix(c(0),p,p,byrow=TRUE)
  for(z in 1:p){
        for(j in 1:p){
        M[z,j]=(   (1/sum(alpha)-alpha[i]/sum(alpha)^2)*cos((i-1)*(ang[j]-ang[z]))  -1*(alpha[1]/sum(alpha)^2+(alpha[-c(1)]/sum(alpha))^2%*%cos(c(1:m)*(ang[j]-ang[z])))     )*Dz[z,z]*Dz[j,j]
        }}
        dp=M
        gradientalph[i]=1*sum(diag(  solve(S)%*%(R-S)%*%solve(S)%*%dp  ))
        Dpalph[,i]=c(dp)
        }
        
        
        for(i in 1:1){
        M=matrix(c(0),p,p,byrow=TRUE)
  for(z in 1:p){
        for(j in 1:p){
        M[z,j]=(  1/sum(alpha)-alpha[i]/sum(alpha)^2 -1*sum( (alpha[-c(1)]/sum(alpha)^2)*cos(c(1:m)*(ang[j]-ang[z]) ) ) )*Dz[z,z]*Dz[j,j]
        }}
dp=M
        gradientalph[i]=1*sum(diag(  solve(S)%*%(R-S)%*%solve(S)%*%dp  ))
        
        Dpalph[,i]=c(dp)
        };Dpalph=Dpalph[,-c(2)] 
}
 if((mcsc=="-1")){      
gradientalph=rep(0,(m+1));Dpalph=matrix(0,p*p,length(alpha))
if(  m>=3 ){ 
for(i in seq(4,m+1,by=2)){
        M=matrix(c(0),p,p,byrow=TRUE)
  for(z in 1:p){
        for(j in 1:p){
        M[z,j]=(   (1/sum(alpha)-alpha[i]/sum(alpha)^2)*cos((i-1)*(ang[j]-ang[z]))  -1*(alpha[1]/sum(alpha)^2+(alpha[-c(1)]/sum(alpha))^2%*%cos(c(1:m)*(ang[j]-ang[z])))     )*Dz[z,z]*Dz[j,j]
        }}
        dp=M
        gradientalph[i]=1*sum(diag(  solve(S)%*%(R-S)%*%solve(S)%*%dp  ))
        Dpalph[,i]=c(dp)
        }
           Dpalph=Dpalph[,-c(2)]}
   
    
if(m<=2){ 
gradientalph=rep(0,(m+1));Dpalph=matrix(0,p*p,length(alpha))
Dpalph=Dpalph[,-c(2)]   }
                 
                  }


if((mcsc=="0")){ 
gradientalph=rep(0,(m+1));Dpalph=matrix(0,p*p,length(alpha))
if(  m>=3 ){ 
for(i in seq(4,m+1,by=2)){
        M=matrix(c(0),p,p,byrow=TRUE)
  for(z in 1:p){
        for(j in 1:p){
        M[z,j]=(  1/sum(alpha) 
        
        - (sum(alpha[seq(2,m+1,by=2)])-sum(alpha[seq(3,m+1,by=2)]))*1/sum(alpha) 
        
        -(sum(alpha[-c(1,i)]*1/sum(alpha)*cos(c(seq(1,m,by=1)[-c(i-1)])*(ang[j]-ang[z]))))  
        
        +(1/sum(alpha)-alpha[i]*1/sum(alpha))*cos(c(i-1)*(ang[j]-ang[z]))   )*Dz[z,z]*Dz[j,j]
        }}
        dp=M
        gradientalph[i]=1*sum(diag(  solve(S)%*%(R-S)%*%solve(S)%*%dp  ))
        Dpalph[,i]=c(dp)
        } 
}
if(  m>=2 ){
for(i in seq(3,m+1,by=2)){
        M=matrix(c(0),p,p,byrow=TRUE)
  for(z in 1:p){
        for(j in 1:p){
        M[z,j]=(  1/sum(alpha)*cos(c(i-1)*(ang[j]-ang[z])) -(1/sum(alpha))  )*Dz[z,z]*Dz[j,j]
        }}
        dp=M
        gradientalph[i]=1*sum(diag(  solve(S)%*%(R-S)%*%solve(S)%*%dp  ))
        Dpalph[,i]=c(dp)
        }
}
  Dpalph=Dpalph[,-c(2)]

}
if( (mcsc=="0" & m==1)){ 
gradientalph=rep(0,(m+1));Dpalph=matrix(0,p*p,length(b))
 Dpalph=Dpalph[,-c(2)]
}


gradientang=rep(0,p);Dpang=matrix(0,p*p,length(ang))
for(i in 1:p){
        M=matrix(c(0),p,p,byrow=TRUE)
  for(z in 1:p){
        for(j in 1:p){
                
                if(z==i) s=1
                if(z!=i) s=0
                if(j==i) sj=1
                if(j!=i) sj=0
        M[z,j]= (s*c(1:m)%*%(b[-c(1)]*sin(c(1:m)*(ang[j]-ang[i]))) - sj*c(1:m)%*%(b[-c(1)]*sin(c(1:m)*(ang[i]-ang[z]))))*Dz[z,z]*Dz[j,j]
        }}
        dp=M;
        Dpang[,i]=c(dp)
        
        gradientang[i]=1*sum(diag(  solve(S)%*%(R-S)%*%solve(S)%*%dp  ))
        
        
        };Dpang=Dpang[,-c(1)];
        gradient=c(gradientang[-c(r)], if((mcsc=="unconstrained") | (mcsc=="-1" & m>=3) | (mcsc=="0" & m>=2)){       
gradientalph[-c(2)]}  else if((mcsc=="-1" & m==1) | (mcsc=="0" & m==1)){0} else if((mcsc=="-1" & m==2)){c(0,0)},gradientv,gradientz)
        if((mcsc=="unconstrained") | (mcsc=="-1" & m>=1) | (mcsc=="0" & m>=1)){
        DerivAnal=data.frame(Dpang,Dpalph,Dpv,Dpz);
        for(i in 1:length(par)){
        for(j in 1:length(par)){
     
                      hessian[i,j]=-1*sum(diag(  solve(S)%*%matrix(DerivAnal[,i],p,p)%*%solve(S)%*%matrix(DerivAnal[,j],p,p) ))

  }}} 
 
 hessian}




objective3max<-
function(par){
    ang=c(start.valuesA(R,k)$polar.angles)*K
      if(mcsc=="unconstrained"){if(m==1){alpha=c(par[(1)],1)
} else alpha=  c(par[(1)],1,par[(2):((m))]);b=alpha/sum(alpha)}
     if(mcsc=="-1" ){if(m==1){b=  c(0,1)} else if(m>=3){alpha=  c(par[(1)],1,par[(2):(m)]);alpha[c(seq(1,(m+1),by=2))]=0; b=alpha/sum(alpha)} else {alpha=  c(par[(1)],1,par[(2):(m)]);alpha[c(seq(1,(m+1),by=2))]=0; b=alpha/sum(alpha)}}
      if(mcsc=="0" ){if(m==1){b=  c(0.5,0.5)} else if(m>=3){alpha=  c(par[(1)],1,par[(2):(m)]);alpha[1]=sum(alpha[seq(2,m+1,by=2)])-sum(alpha[seq(3,m+1,by=2)]); b=alpha/sum(alpha)} else {alpha=  c(par[(1)],1,par[(2):(m)]);alpha[1]=sum(alpha[seq(2,m+1,by=2)])-alpha[3]; b=alpha/sum(alpha)}}

        


         v=  par[((m)+1)]
                                   z=  par[((m)+1+1):((m)+1+p)]

K=pi/180
M=matrix(c(0),p,p,byrow=TRUE)
  for(i in 1:p){
        for(j in 1:p){
        M[i,j]=c(b[-c(1)])%*%cos(c(1:m)*(ang[j]-ang[i]))
        }}
 Pc=M+matrix(b[1],p,p)
 Dv=diag(v,p)
 Dz=diag(z); f=-1*(sum(diag(R%*%solve(Dz%*%(Pc+Dv)%*%Dz)))+log(det(Dz%*%(Pc+Dv)%*%Dz))-log(det(R))-p) ;
 S1=Dz%*%(Pc+Dv)%*%Dz;S2=diag(1/sqrt(diag(S1)))%*%Dz;S=S2%*%(Pc+Dv)%*%S2
 attributes(f)=list(f=f,S=S,Cs=Dz%*%(Pc+Dv)%*%Dz,Pc=Pc)

 f}
 
objective3gr<-
function(par){
    ang=c(start.valuesA(R,k)$polar.angles)*K
      if(mcsc=="unconstrained"){if(m==1){alpha=c(par[(1)],1)
} else alpha=  c(par[(1)],1,par[(2):((m))]);b=alpha/sum(alpha)}
     if(mcsc=="-1" ){if(m==1){b=alpha=  c(0,1)} else if(m>=3){alpha=  c(par[(1)],1,par[(2):(m)]);alpha[c(seq(1,(m+1),by=2))]=0; b=alpha/sum(alpha)} else {alpha=  c(par[(1)],1,par[(2):(m)]);alpha[c(seq(1,(m+1),by=2))]=0; b=alpha/sum(alpha)}}
      if(mcsc=="0" ){if(m==1){b=alpha=  c(0.5,0.5)} else if(m>=3){alpha=  c(par[(1)],1,par[(2):(m)]);alpha[1]=sum(alpha[seq(2,m+1,by=2)])-sum(alpha[seq(3,m+1,by=2)]); b=alpha/sum(alpha)} else {alpha=  c(par[(1)],1,par[(2):(m)]);alpha[1]=sum(alpha[seq(2,m+1,by=2)])-alpha[3]; b=alpha/sum(alpha)}}

         v=  par[((m)+1)]
                                   z=  par[((m)+1+1):((m)+1+p)]

K=pi/180

M=matrix(c(0),p,p,byrow=TRUE)
  for(i in 1:p){
        for(j in 1:p){
        M[i,j]=c(b[-c(1)])%*%cos(c(1:m)*(ang[j]-ang[i]))
        }}
 Pc=M+matrix(b[1],p,p)
 Dv=diag(v,p)
 Dz=diag(z); 
 S=Dz%*%(Pc+Dv)%*%Dz


 hessian=matrix(0,length(par),length(par))

 gradientz=rep(0,p);Dpz=matrix(0,p*p,p)
for(i in 1:p){
        J=matrix(0,p,p)
        J[i,i]=1;
        dp=J%*%(Pc+Dv)%*%Dz+Dz%*%(Pc+Dv)%*%J
        gradientz[i]=1*sum(diag(  solve(S)%*%(R-S)%*%solve(S)%*%dp  ))
        Dpz[,i]=c(dp)
        }
gradientv=rep(0,p);Dpv=matrix(0,p*p,1)
        J=diag(1,p,p) 
        dp=J%*%(Dz)^2
        gradientv=1*sum(diag(  solve(S)%*%(R-S)%*%solve(S)%*%dp  ))
        Dpv=c(dp)

 if((mcsc=="unconstrained")){       
gradientalph=rep(0,(m+1));Dpalph=matrix(0,p*p,length(alpha))
for(i in 2:(m+1)){
        M=matrix(c(0),p,p,byrow=TRUE)
  for(z in 1:p){
        for(j in 1:p){
        M[z,j]=(    (1/sum(alpha)-alpha[i]/sum(alpha)^2)*cos((i-1)*(ang[j]-ang[z]))  -1*(alpha[1]/sum(alpha)^2+(alpha[-c(1)]/sum(alpha))^2%*%cos(c(1:m)*(ang[j]-ang[z])))       )*Dz[z,z]*Dz[j,j] 
        }}
        dp=M
        gradientalph[i]=1*sum(diag(  solve(S)%*%(R-S)%*%solve(S)%*%dp  ))
        Dpalph[,i]=c(dp)
        }
        
        
        for(i in 1:1){
        M=matrix(c(0),p,p,byrow=TRUE)
  for(z in 1:p){
        for(j in 1:p){
        M[z,j]=(  1/sum(alpha)-alpha[i]/sum(alpha)^2 -1*sum( (alpha[-c(1)]/sum(alpha)^2)*cos(c(1:m)*(ang[j]-ang[z]) ) ) )*Dz[z,z]*Dz[j,j]
        }}
dp=M
        gradientalph[i]=1*sum(diag(  solve(S)%*%(R-S)%*%solve(S)%*%dp  ))
        
        Dpalph[,i]=c(dp)
        }  
}

 if((mcsc=="-1")){      
gradientalph=rep(0,(m+1));Dpalph=matrix(0,p*p,length(alpha))
if(  m>=3 ){ 
for(i in seq(4,m+1,by=2)){
        M=matrix(c(0),p,p,byrow=TRUE)
  for(z in 1:p){
        for(j in 1:p){
        M[z,j]=(   (1/sum(alpha)-alpha[i]/sum(alpha)^2)*cos((i-1)*(ang[j]-ang[z]))  -1*(alpha[1]/sum(alpha)^2+(alpha[-c(1)]/sum(alpha))^2%*%cos(c(1:m)*(ang[j]-ang[z])))      )*Dz[z,z]*Dz[j,j]  
        }}
        dp=M
        gradientalph[i]=1*sum(diag(  solve(S)%*%(R-S)%*%solve(S)%*%dp  ))
        Dpalph[,i]=c(dp)
        }
           }
   
if(( m<=2)){ 
gradientalph=rep(0,(m+1));Dpalph=matrix(0,p*p,length(alpha))
}
        
                 }


if((mcsc=="0")){ 
gradientalph=rep(0,(m+1));Dpalph=matrix(0,p*p,length(alpha))
if(  m>=3 ){ 
for(i in seq(4,m+1,by=2)){
        M=matrix(c(0),p,p,byrow=TRUE)
  for(z in 1:p){
        for(j in 1:p){
        M[z,j]=(  1/sum(alpha) 
        
        - (sum(alpha[seq(2,m+1,by=2)])-sum(alpha[seq(3,m+1,by=2)]))*1/sum(alpha) 
        
        -(sum(alpha[-c(1,i)]*1/sum(alpha)*cos(c(seq(1,m,by=1)[-c(i-1)])*(ang[j]-ang[z]))))  
        
        +(1/sum(alpha)-alpha[i]*1/sum(alpha))*cos(c(i-1)*(ang[j]-ang[z]))   )*Dz[z,z]*Dz[j,j]
        }}
        dp=M
        gradientalph[i]=1*sum(diag(  solve(S)%*%(R-S)%*%solve(S)%*%dp  ))
        Dpalph[,i]=c(dp)
        } 
}
if(  m>=2 ){ 
for(i in seq(3,m+1,by=2)){
        M=matrix(c(0),p,p,byrow=TRUE)
  for(z in 1:p){
        for(j in 1:p){
        M[z,j]=(  1/sum(alpha)*cos(c(i-1)*(ang[j]-ang[z])) -(1/sum(alpha))  )*Dz[z,z]*Dz[j,j]
        }}
        dp=M
        gradientalph[i]=1*sum(diag(  solve(S)%*%(R-S)%*%solve(S)%*%dp  ))
        Dpalph[,i]=c(dp)
        }
}
  
}
if((mcsc=="0" & m==1)){ 
gradientalph=rep(0,(m+1));Dpalph=matrix(0,p*p,length(alpha))
}

gradient=c( if((mcsc=="unconstrained") | (mcsc=="-1" & m>=3) | (mcsc=="0" & m>=2)){       
gradientalph[-c(2)]} else if((mcsc=="-1" & m==1) | (mcsc=="0" & m==1)){0} else if((mcsc=="-1" & m==2)){c(0,0)},gradientv,gradientz)        

 
 gradient}

objective3hess<-
function(par){
    ang=c(start.valuesA(R,k)$polar.angles)*K
      if(mcsc=="unconstrained"){if(m==1){alpha=c(par[(1)],1)
} else alpha=  c(par[(1)],1,par[(2):((m))]);b=alpha/sum(alpha)}
     if(mcsc=="-1" ){if(m==1){b=alpha=  c(0,1)} else if(m>=3){alpha=  c(par[(1)],1,par[(2):(m)]);alpha[c(seq(1,(m+1),by=2))]=0; b=alpha/sum(alpha)} else {alpha=  c(par[(1)],1,par[(2):(m)]);alpha[c(seq(1,(m+1),by=2))]=0; b=alpha/sum(alpha)}}
      if(mcsc=="0" ){if(m==1){b=alpha=  c(0.5,0.5)} else if(m>=3){alpha=  c(par[(1)],1,par[(2):(m)]);alpha[1]=sum(alpha[seq(2,m+1,by=2)])-sum(alpha[seq(3,m+1,by=2)]); b=alpha/sum(alpha)} else {alpha=  c(par[(1)],1,par[(2):(m)]);alpha[1]=sum(alpha[seq(2,m+1,by=2)])-alpha[3]; b=alpha/sum(alpha)}}

         v=  par[((m)+1)]
                                   z=  par[((m)+1+1):((m)+1+p)]

K=pi/180
M=matrix(c(0),p,p,byrow=TRUE)
  for(i in 1:p){
        for(j in 1:p){
        M[i,j]=c(b[-c(1)])%*%cos(c(1:m)*(ang[j]-ang[i]))
        }}
 Pc=M+matrix(b[1],p,p)
 Dv=diag(v,p)
 Dz=diag(z); 

 S=Dz%*%(Pc+Dv)%*%Dz;S2=diag(1/sqrt(diag(S)))%*%Dz;S1=S2%*%(Pc+Dv)%*%S2
 

 hessian=matrix(0,length(par),length(par))

 gradientz=rep(0,p);Dpz=matrix(0,p*p,p)
for(i in 1:p){
        J=matrix(0,p,p)
        J[i,i]=1;
        dp=J%*%(Pc+Dv)%*%Dz+Dz%*%(Pc+Dv)%*%J
        gradientz[i]=1*sum(diag(  solve(S)%*%(R-S)%*%solve(S)%*%dp  ))
        Dpz[,i]=c(dp)
        }
gradientv=rep(0,p);Dpv=matrix(0,p*p,1)
        J=diag(1,p,p) 
        dp=J%*%(Dz)^2
        gradientv=1*sum(diag(  solve(S)%*%(R-S)%*%solve(S)%*%dp  ))
        Dpv=c(dp)

 if((mcsc=="unconstrained")){       
gradientalph=rep(0,(m+1));Dpalph=matrix(0,p*p,length(alpha))
for(i in 2:(m+1)){
        M=matrix(c(0),p,p,byrow=TRUE)
  for(z in 1:p){
        for(j in 1:p){
        M[z,j]=(    (1/sum(alpha)-alpha[i]/sum(alpha)^2)*cos((i-1)*(ang[j]-ang[z]))  -1*(alpha[1]/sum(alpha)^2+(alpha[-c(1)]/sum(alpha))^2%*%cos(c(1:m)*(ang[j]-ang[z])))       )*Dz[z,z]*Dz[j,j] 
        }}
        dp=M
        gradientalph[i]=1*sum(diag(  solve(S)%*%(R-S)%*%solve(S)%*%dp  ))
        Dpalph[,i]=c(dp)
        }
        
        
        for(i in 1:1){
        M=matrix(c(0),p,p,byrow=TRUE)
  for(z in 1:p){
        for(j in 1:p){
        M[z,j]=(  1/sum(alpha)-alpha[i]/sum(alpha)^2 -1*sum( (alpha[-c(1)]/sum(alpha)^2)*cos(c(1:m)*(ang[j]-ang[z]) ) ) )*Dz[z,z]*Dz[j,j]
        }}
dp=M
        gradientalph[i]=1*sum(diag(  solve(S)%*%(R-S)%*%solve(S)%*%dp  ))
        
        Dpalph[,i]=c(dp)
        };if(mcsc=="unconstrained"){Dpalph=Dpalph[,-c(2)]} ;  
}

 if((mcsc=="-1")){      
gradientalph=rep(0,(m+1));Dpalph=matrix(0,p*p,length(alpha))
if(  m>=3 ){ 
for(i in seq(4,m+1,by=2)){
        M=matrix(c(0),p,p,byrow=TRUE)
  for(z in 1:p){
        for(j in 1:p){
        M[z,j]=(   (1/sum(alpha)-alpha[i]/sum(alpha)^2)*cos((i-1)*(ang[j]-ang[z]))  -1*(alpha[1]/sum(alpha)^2+(alpha[-c(1)]/sum(alpha))^2%*%cos(c(1:m)*(ang[j]-ang[z])))      )*Dz[z,z]*Dz[j,j]  
        }}
        dp=M
        gradientalph[i]=1*sum(diag(  solve(S)%*%(R-S)%*%solve(S)%*%dp  ))
        Dpalph[,i]=c(dp)
        }
           };Dpalph=Dpalph[,-c(2)]
   
        
                 }
if((mcsc=="-1" & m<=2)){ 
gradientalph=rep(0,(m+1));Dpalph=matrix(0,p*p,length(alpha))
Dpalph=Dpalph[,-c(2)]}

if((mcsc=="0")){ 
gradientalph=rep(0,(m+1));Dpalph=matrix(0,p*p,length(alpha))
if(  m>=3 ){ 
for(i in seq(4,m+1,by=2)){
        M=matrix(c(0),p,p,byrow=TRUE)
  for(z in 1:p){
        for(j in 1:p){
        M[z,j]=(  1/sum(alpha) 
        
        - (sum(alpha[seq(2,m+1,by=2)])-sum(alpha[seq(3,m+1,by=2)]))*1/sum(alpha) 
        
        -(sum(alpha[-c(1,i)]*1/sum(alpha)*cos(c(seq(1,m,by=1)[-c(i-1)])*(ang[j]-ang[z]))))  
        
        +(1/sum(alpha)-alpha[i]*1/sum(alpha))*cos(c(i-1)*(ang[j]-ang[z]))   )*Dz[z,z]*Dz[j,j]
        }}
        dp=M
        gradientalph[i]=1*sum(diag(  solve(S)%*%(R-S)%*%solve(S)%*%dp  ))
        Dpalph[,i]=c(dp)
        } 
}
if(  m>=2 ){ 
for(i in seq(3,m+1,by=2)){
        M=matrix(c(0),p,p,byrow=TRUE)
  for(z in 1:p){
        for(j in 1:p){
        M[z,j]=(  1/sum(alpha)*cos(c(i-1)*(ang[j]-ang[z])) -(1/sum(alpha))  )*Dz[z,z]*Dz[j,j]
        }}
        dp=M
        gradientalph[i]=1*sum(diag(  solve(S)%*%(R-S)%*%solve(S)%*%dp  ))
        Dpalph[,i]=c(dp)
        }
}
  Dpalph=Dpalph[,-c(2)]
}

if((mcsc=="0" & m==1)){ 
gradientalph=rep(0,(m+1));Dpalph=matrix(0,p*p,length(alpha))
 Dpalph=Dpalph[,-c(2)]
}

        gradient=c( if((mcsc=="unconstrained") | (mcsc=="-1" & m>=3) | (mcsc=="0" & m>=2)){       
gradientalph[-c(2)]}  else if((mcsc=="-1" & m==1) | (mcsc=="0" & m==1)){0} else if((mcsc=="-1" & m==2)){c(0,0)},gradientv,gradientz)
        if((mcsc=="unconstrained") | (mcsc=="-1" & m>=1) | (mcsc=="0" & m>=1)){
        DerivAnal=data.frame(Dpalph,Dpv,Dpz);
        for(i in 1:length(par)){
        for(j in 1:length(par)){
     
                      hessian[i,j]=-1*sum(diag(  solve(S)%*%matrix(DerivAnal[,i],p,p)%*%solve(S)%*%matrix(DerivAnal[,j],p,p) ))
 
  }}} 
 
 hessian}
 



objective4max=
function(par){
    ang=c(start.valuesA(R,k)$polar.angles)*K
      if(mcsc=="unconstrained"){if(m==1){alpha=  c(par[(1)],1)} else alpha=  c(par[(1)],1,par[(2):((m))]);b=alpha/sum(alpha)}
      if(mcsc=="-1" ){if(m==1){b=  c(0,1)} else if(m>=3){alpha=  c(par[(1)],1,par[(2):(m)]);alpha[c(seq(1,(m+1),by=2))]=0; b=alpha/sum(alpha)} else {alpha=  c(par[(1)],1,par[(2):(m)]);alpha[c(seq(1,(m+1),by=2))]=0; b=alpha/sum(alpha)}}
      if(mcsc=="0" ){if(m==1){b=c(0.5,0.5)} else if(m>=3){alpha=  c(par[(1)],1,par[(2):(m)]);alpha[1]=sum(alpha[seq(2,m+1,by=2)])-sum(alpha[seq(3,m+1,by=2)]); b=alpha/sum(alpha)} else {alpha=  c(par[(1)],1,par[(2):(m)]);alpha[1]=sum(alpha[seq(2,m+1,by=2)])-alpha[3]; b=alpha/sum(alpha)}}

         v=  par[((m)+1):((m)+p)]
                                   z=  par[((m)+p+1):((m)+p+p)]

 K=pi/180
 M=matrix(c(0),p,p,byrow=TRUE)
  for(i in 1:p){
        for(j in 1:p){
        M[i,j]=c(b[-c(1)])%*%cos(c(1:m)*(ang[j]-ang[i]))
        }}
 Pc=M+matrix(b[1],p,p)
 Dv=diag(v,p)
 Dz=diag(z); f=-1*(sum(diag(R%*%solve(Dz%*%(Pc+Dv)%*%Dz)))+log(det(Dz%*%(Pc+Dv)%*%Dz))-log(det(R))-p) ;
 S1=Dz%*%(Pc+Dv)%*%Dz;S2=diag(1/sqrt(diag(S1)))%*%Dz;S=S2%*%(Pc+Dv)%*%S2
 attributes(f)=list(f=f,S=S,Cs=Dz%*%(Pc+Dv)%*%Dz,Pc=Pc)
 f}
 
 objective4gr<-
function(par){
          ang=c(start.valuesA(R,k)$polar.angles)*K
      if(mcsc=="unconstrained"){if(m==1){alpha=  c(par[(1)],1)} else alpha=  c(par[(1)],1,par[(2):((m))]);b=alpha/sum(alpha)}
     if(mcsc=="-1" ){if(m==1){b=alpha=  c(0,1)} else if(m>=3){alpha=  c(par[(1)],1,par[(2):(m)]);alpha[c(seq(1,(m+1),by=2))]=0; b=alpha/sum(alpha)} else {alpha=  c(par[(1)],1,par[(2):(m)]);alpha[c(seq(1,(m+1),by=2))]=0; b=alpha/sum(alpha)}}
      if(mcsc=="0" ){if(m==1){b=alpha=  c(0.5,0.5)} else if(m>=3){alpha=  c(par[(1)],1,par[(2):(m)]);alpha[1]=sum(alpha[seq(2,m+1,by=2)])-sum(alpha[seq(3,m+1,by=2)]); b=alpha/sum(alpha)} else {alpha=  c(par[(1)],1,par[(2):(m)]);alpha[1]=sum(alpha[seq(2,m+1,by=2)])-alpha[3]; b=alpha/sum(alpha)}}
         v=  par[((m)+1):((m)+p)]
                                   z=  par[((m)+p+1):((m)+p+p)]

 K=pi/180
 M=matrix(c(0),p,p,byrow=TRUE)
  for(i in 1:p){
        for(j in 1:p){
        M[i,j]=c(b[-c(1)])%*%cos(c(1:m)*(ang[j]-ang[i]))
        }}
 Pc=M+matrix(b[1],p,p)
 Dv=diag(v,p)
 Dz=diag(z); 
 S=Dz%*%(Pc+Dv)%*%Dz
 

 hessian=matrix(0,length(par),length(par))

 gradientz=rep(0,p);Dpz=matrix(0,p*p,p)
for(i in 1:p){
        J=matrix(0,p,p)
        J[i,i]=1;
        dp=J%*%(Pc+Dv)%*%Dz+Dz%*%(Pc+Dv)%*%J
        gradientz[i]=1*sum(diag(  solve(S)%*%(R-S)%*%solve(S)%*%dp  ))
        Dpz[,i]=c(dp)
        }
gradientv=rep(0,p);Dpv=matrix(0,p*p,p)
for(i in 1:p){
        J=matrix(0,p,p)
        J[i,i]=1; 
        dp=J%*%(Dz)^2
        gradientv[i]=1*sum(diag(  solve(S)%*%(R-S)%*%solve(S)%*%dp  ))
        Dpv[,i]=c(dp)
        }

 if((mcsc=="unconstrained")){       
gradientalph=rep(0,(m+1));Dpalph=matrix(0,p*p,length(alpha))
for(i in 2:(m+1)){
        M=matrix(c(0),p,p,byrow=TRUE)
  for(z in 1:p){
        for(j in 1:p){
        M[z,j]=(   (1/sum(alpha)-alpha[i]/sum(alpha)^2)*cos((i-1)*(ang[j]-ang[z]))  -1*(alpha[1]/sum(alpha)^2+(alpha[-c(1)]/sum(alpha))^2%*%cos(c(1:m)*(ang[j]-ang[z])))     )*Dz[z,z]*Dz[j,j]
        }}
        dp=M
        gradientalph[i]=1*sum(diag(  solve(S)%*%(R-S)%*%solve(S)%*%dp  ))
        Dpalph[,i]=c(dp)
        }
        
        
        for(i in 1:1){
        M=matrix(c(0),p,p,byrow=TRUE)
  for(z in 1:p){
        for(j in 1:p){
        M[z,j]=(  1/sum(alpha)-alpha[i]/sum(alpha)^2 -1*sum( (alpha[-c(1)]/sum(alpha)^2)*cos(c(1:m)*(ang[j]-ang[z]) ) ) )*Dz[z,z]*Dz[j,j]
        }}
dp=M
        gradientalph[i]=1*sum(diag(  solve(S)%*%(R-S)%*%solve(S)%*%dp  ))
        
        Dpalph[,i]=c(dp)
        }  
}

 if((mcsc=="-1")){      
gradientalph=rep(0,(m+1));Dpalph=matrix(0,p*p,length(alpha))
if(  m>=3 ){ 
for(i in seq(4,m+1,by=2)){
        M=matrix(c(0),p,p,byrow=TRUE)
  for(z in 1:p){
        for(j in 1:p){
        M[z,j]=(    (1/sum(alpha)-alpha[i]/sum(alpha)^2)*cos((i-1)*(ang[j]-ang[z]))  -1*(alpha[1]/sum(alpha)^2+(alpha[-c(1)]/sum(alpha))^2%*%cos(c(1:m)*(ang[j]-ang[z])))    )*Dz[z,z]*Dz[j,j]
        }}
        dp=M
        gradientalph[i]=1*sum(diag(  solve(S)%*%(R-S)%*%solve(S)%*%dp  ))
        Dpalph[,i]=c(dp)
        }
           }
   
        
                 }
   
        
                 
if((mcsc=="-1" & m<=2)){ 
gradientalph=rep(0,(m+1));Dpalph=matrix(0,p*p,length(alpha))
}
if((mcsc=="0")){ 
gradientalph=rep(0,(m+1));Dpalph=matrix(0,p*p,length(alpha))
if(  m>=3 ){ 
for(i in seq(4,m+1,by=2)){
        M=matrix(c(0),p,p,byrow=TRUE)
  for(z in 1:p){
        for(j in 1:p){
        M[z,j]=(  1/sum(alpha) 
        
        - (sum(alpha[seq(2,m+1,by=2)])-sum(alpha[seq(3,m+1,by=2)]))*1/sum(alpha) 
        
        -(sum(alpha[-c(1,i)]*1/sum(alpha)*cos(c(seq(1,m,by=1)[-c(i-1)])*(ang[j]-ang[z]))))  
        
        +(1/sum(alpha)-alpha[i]*1/sum(alpha))*cos(c(i-1)*(ang[j]-ang[z]))   )*Dz[z,z]*Dz[j,j]
        }}
        dp=M
        gradientalph[i]=1*sum(diag(  solve(S)%*%(R-S)%*%solve(S)%*%dp  ))
        Dpalph[,i]=c(dp)
        } 
}
if(  m>=2 ){ 
for(i in seq(3,m+1,by=2)){
        M=matrix(c(0),p,p,byrow=TRUE)
  for(z in 1:p){
        for(j in 1:p){
        M[z,j]=(  1/sum(alpha)*cos(c(i-1)*(ang[j]-ang[z])) -(1/sum(alpha))  )*Dz[z,z]*Dz[j,j]
        }}
        dp=M
        gradientalph[i]=1*sum(diag(  solve(S)%*%(R-S)%*%solve(S)%*%dp  ))
        Dpalph[,i]=c(dp)
        }
}
  
}
if((mcsc=="0" & m==1)){ 
gradientalph=rep(0,(m+1));Dpalph=matrix(0,p*p,length(alpha))
 
}
gradient=c( if((mcsc=="unconstrained") | (mcsc=="-1" & m>=3) | (mcsc=="0" & m>=2)){       
gradientalph[-c(2)]} else if((mcsc=="-1" & m==1) | (mcsc=="0" & m==1)){0} else if((mcsc=="-1" & m==2)){c(0,0)},gradientv,gradientz)        

 
 gradient}
 
objective4hess<-
function(par){
          ang=c(start.valuesA(R,k)$polar.angles)*K
      if(mcsc=="unconstrained"){if(m==1){alpha=  c(par[(1)],1)} else alpha=  c(par[(1)],1,par[(2):((m))]);b=alpha/sum(alpha)}
       if(mcsc=="-1" ){if(m==1){b=alpha=  c(0,1)} else if(m>=3){alpha=  c(par[(1)],1,par[(2):(m)]);alpha[c(seq(1,(m+1),by=2))]=0; b=alpha/sum(alpha)} else {alpha=  c(par[(1)],1,par[(2):(m)]);alpha[c(seq(1,(m+1),by=2))]=0; b=alpha/sum(alpha)}}
      if(mcsc=="0" ){if(m==1){b=alpha=  c(0.5,0.5)} else if(m>=3){alpha=  c(par[(1)],1,par[(2):(m)]);alpha[1]=sum(alpha[seq(2,m+1,by=2)])-sum(alpha[seq(3,m+1,by=2)]); b=alpha/sum(alpha)} else {alpha=  c(par[(1)],1,par[(2):(m)]);alpha[1]=sum(alpha[seq(2,m+1,by=2)])-alpha[3]; b=alpha/sum(alpha)}}
         v=  par[((m)+1):((m)+p)]
                                   z=  par[((m)+p+1):((m)+p+p)]


 K=pi/180
 M=matrix(c(0),p,p,byrow=TRUE)
  for(i in 1:p){
        for(j in 1:p){
        M[i,j]=c(b[-c(1)])%*%cos(c(1:m)*(ang[j]-ang[i]))
        }}
 Pc=M+matrix(b[1],p,p)
 Dv=diag(v,p)
 Dz=diag(z); 
 
 S=Dz%*%(Pc+Dv)%*%Dz;S2=diag(1/sqrt(diag(S)))%*%Dz;S1=S2%*%(Pc+Dv)%*%S2
 

 hessian=matrix(0,length(par),length(par))

 gradientz=rep(0,p);Dpz=matrix(0,p*p,p)
for(i in 1:p){
        J=matrix(0,p,p)
        J[i,i]=1;
        dp=J%*%(Pc+Dv)%*%Dz+Dz%*%(Pc+Dv)%*%J
        gradientz[i]=1*sum(diag(  solve(S)%*%(R-S)%*%solve(S)%*%dp  ))
        Dpz[,i]=c(dp)
        }
gradientv=rep(0,p);Dpv=matrix(0,p*p,p)
for(i in 1:p){
        J=matrix(0,p,p)
        J[i,i]=1; 
        dp=J%*%(Dz)^2
        gradientv[i]=1*sum(diag(  solve(S)%*%(R-S)%*%solve(S)%*%dp  ))
        Dpv[,i]=c(dp)
        }

 if((mcsc=="unconstrained")){       
gradientalph=rep(0,(m+1));Dpalph=matrix(0,p*p,length(alpha))
for(i in 2:(m+1)){
        M=matrix(c(0),p,p,byrow=TRUE)
  for(z in 1:p){
        for(j in 1:p){
        M[z,j]=(   (1/sum(alpha)-alpha[i]/sum(alpha)^2)*cos((i-1)*(ang[j]-ang[z]))  -1*(alpha[1]/sum(alpha)^2+(alpha[-c(1)]/sum(alpha))^2%*%cos(c(1:m)*(ang[j]-ang[z])))     )*Dz[z,z]*Dz[j,j]
        }}
        dp=M
        gradientalph[i]=1*sum(diag(  solve(S)%*%(R-S)%*%solve(S)%*%dp  ))
        Dpalph[,i]=c(dp)
        }
        
        
        for(i in 1:1){
        M=matrix(c(0),p,p,byrow=TRUE)
  for(z in 1:p){
        for(j in 1:p){
        M[z,j]=(  1/sum(alpha)-alpha[i]/sum(alpha)^2 -1*sum( (alpha[-c(1)]/sum(alpha)^2)*cos(c(1:m)*(ang[j]-ang[z]) ) ) )*Dz[z,z]*Dz[j,j]
        }}
dp=M
        gradientalph[i]=1*sum(diag(  solve(S)%*%(R-S)%*%solve(S)%*%dp  ))
        
        Dpalph[,i]=c(dp)
        };if(mcsc=="unconstrained"){Dpalph=Dpalph[,-c(2)]} ;  
}

 if((mcsc=="-1")){      
gradientalph=rep(0,(m+1));Dpalph=matrix(0,p*p,length(alpha))
if(  m>=3 ){ 
for(i in seq(4,m+1,by=2)){
        M=matrix(c(0),p,p,byrow=TRUE)
  for(z in 1:p){
        for(j in 1:p){
        M[z,j]=(    (1/sum(alpha)-alpha[i]/sum(alpha)^2)*cos((i-1)*(ang[j]-ang[z]))  -1*(alpha[1]/sum(alpha)^2+(alpha[-c(1)]/sum(alpha))^2%*%cos(c(1:m)*(ang[j]-ang[z])))    )*Dz[z,z]*Dz[j,j]
        }}
        dp=M
        gradientalph[i]=1*sum(diag(  solve(S)%*%(R-S)%*%solve(S)%*%dp  ))
        Dpalph[,i]=c(dp)
        }
           };Dpalph=Dpalph[,-c(2)]
   
        
                 }
   
        
                 
if((mcsc=="-1" & m<=2)){ 
gradientalph=rep(0,(m+1));Dpalph=matrix(0,p*p,length(alpha))
Dpalph=Dpalph[,-c(2)]}

if((mcsc=="0")){ 
gradientalph=rep(0,(m+1));Dpalph=matrix(0,p*p,length(alpha))
if(  m>=3 ){ 
for(i in seq(4,m+1,by=2)){
        M=matrix(c(0),p,p,byrow=TRUE)
  for(z in 1:p){
        for(j in 1:p){
        M[z,j]=(  1/sum(alpha) 
        
        - (sum(alpha[seq(2,m+1,by=2)])-sum(alpha[seq(3,m+1,by=2)]))*1/sum(alpha) 
        
        -(sum(alpha[-c(1,i)]*1/sum(alpha)*cos(c(seq(1,m,by=1)[-c(i-1)])*(ang[j]-ang[z]))))  
        
        +(1/sum(alpha)-alpha[i]*1/sum(alpha))*cos(c(i-1)*(ang[j]-ang[z]))   )*Dz[z,z]*Dz[j,j]
        }}
        dp=M
        gradientalph[i]=1*sum(diag(  solve(S)%*%(R-S)%*%solve(S)%*%dp  ))
        Dpalph[,i]=c(dp)
        } 
}
if(  m>=2 ){ 
for(i in seq(3,m+1,by=2)){
        M=matrix(c(0),p,p,byrow=TRUE)
  for(z in 1:p){
        for(j in 1:p){
        M[z,j]=(  1/sum(alpha)*cos(c(i-1)*(ang[j]-ang[z])) -(1/sum(alpha))  )*Dz[z,z]*Dz[j,j]
        }}
        dp=M
        gradientalph[i]=1*sum(diag(  solve(S)%*%(R-S)%*%solve(S)%*%dp  ))
        Dpalph[,i]=c(dp)
        }
}
if((mcsc=="0" & m==1)){ 
gradientalph=rep(0,(m+1));Dpalph=matrix(0,p*p,length(alpha))
}

  Dpalph=Dpalph[,-c(2)]
}

gradient=c( if((mcsc=="unconstrained") | (mcsc=="-1" & m>=3) | (mcsc=="0" & m>=2)){       
gradientalph[-c(2)]} else if((mcsc=="-1" & m==1) | (mcsc=="0" & m==1)){0} else if((mcsc=="-1" & m==2)){c(0,0)},gradientv,gradientz)        
        if((mcsc=="unconstrained") | (mcsc=="-1" & m>=1) | (mcsc=="0" & m>=1)){
        DerivAnal=data.frame(Dpalph,Dpv,Dpz);
        for(i in 1:length(par)){
        for(j in 1:length(par)){
     
                      hessian[i,j]=-1*sum(diag(  solve(S)%*%matrix(DerivAnal[,i],p,p)%*%solve(S)%*%matrix(DerivAnal[,j],p,p) ))

  }}}
 
 hessian}
 
 
 
 numericGradient <- function(f, t0, eps=1e-6, ...) {
   NPar <- length(t0)
   NVal <- length(f0 <- f(t0, ...))
   grad <- matrix(NA, NVal, NPar)
   row.names(grad) <- names(f0)
   colnames(grad) <- names(t0)
   for(i in 1:NPar) {
      t2 <- t1 <- t0
      t1[i] <- t0[i] - eps/2
      t2[i] <- t0[i] + eps/2
      grad[,i] <- (f(t2, ...) - f(t1, ...))/eps
   }
   return(grad)
}
 



maxL.BFGS.B <- function(fn, grad=NULL,
                    start,up,low,...) {
                    factr
                    pgtol
                    print.level
                    iterlim
                    
   message <- function(c) {
      switch(as.character(c),
               "0" = "successful convergence",
               "10" = "degeneracy in Nelder-Mead simplex"
               )
   };
if(equal.ang==FALSE ){
	if(m==1){par.names=c(v.names[-c(1)],paste(rep("a",m),c(0)),if(equal.com==TRUE) rep("v",1) else  paste(rep("v",p),v.names),paste(rep("z",p),v.names))}
 else par.names=c(v.names[-c(1)],paste(rep("a",m),c(0,2:m)),if(equal.com==TRUE) rep("v",1) else  paste(rep("v",p),v.names),paste(rep("z",p),v.names))}
   
if(equal.ang==TRUE ){
	if(m==1){par.names=c(paste(rep("a",m),c(0)),if(equal.com==TRUE) rep("v",1) else  paste(rep("v",p),v.names),paste(rep("z",p),v.names))} else par.names=c(paste(rep("a",m),c(0,2:m)),if(equal.com==TRUE) rep("v",1) else  paste(rep("v",p),v.names),paste(rep("z",p),v.names))}



   nParam <- length(start)
   func <- function(theta, ...) {
      sum(fn(theta, ...))
   }
   gradient <- function(theta, ...) {
      if(!is.null(grad)) {
         g <- grad(theta, ...)
         if(!is.null(dim(g))) {
            if(ncol(g) > 1) {
               return(colSums(g))
            }
         } else {
            return(g)
         }
      }
      g <- numericGradient(fn, theta, ...)
      if(!is.null(dim(g))) {
         return(colSums(g))
      } else {
         return(g)
      }
   }
   G1 <- gradient(start, ...)
   if(any(is.na(G1))) {
      stop("Na in the initial gradient")
   }
   if(length(G1) != nParam) {
      stop( "length of gradient (", length(G1),
         ") not equal to the no. of parameters (", nParam, ")" )
   }
   
   
   if( print.level == 1 ) {
   	  cat( "    -------------------------------\n")
      cat( "          Initial parameters:      \n")
      cat( "    -------------------------------\n")
      a <- cbind(start, G1, up,low)
      dimnames(a) <- list(par.names, c("parameter", "initial gradient",
                                          "upper","lower"))
      print(a)
      cat( "                               \n")
      cat( "                               \n")
      cat( "   Constrained (L-BFGS-B) Optimization\n")

         }

   type <- "L-BFGS-B maximisation"
   control <- list(trace=print.level,
                    REPORT=1,
                    fnscale=-1,
                    abstol=1e6,
                    maxit=iterlim,
                    factr=factr,
                    pgtol=pgtol,
                    lmm=if(is.null(lmm)){length(parA)} else lmm)
   a <- optim(start, func, gr=gradient, control=control, method="L-BFGS-B",
      hessian=FALSE,upper=up,lower=low, ...)


Active.Bounds.Up<-which(a$par==up)
Active.Bounds.Low<-which(a$par==low)
{if(length(Active.Bounds.Up)!=0 | length(Active.Bounds.Low)!=0){
Active.Bounds<-c(if(length(Active.Bounds.Up)!=0)Active.Bounds.Up,if(length(Active.Bounds.Low)!=0)Active.Bounds.Low)
Active.Bounds.names<-par.names[Active.Bounds]}
else Active.Bounds.names<-NULL}



   result <- list(
                   maximum=a$value,
                   estimate=a$par,
                   gradient=gradient(a$par),
                   hessian=a$hessian,
                   code=a$convergence,
                   message=paste(message(a$convergence), a$message),
                   last.step=NULL,
                   iterations=a$counts,
                   type=type,
                   Active.Bounds.names=Active.Bounds.names)
   class(result) <- "maximisation"
   invisible(result)
}


 if(is.vector(upper)) {up=c(upper[-c(1)]*K,rep(Inf,length(parA[-c(1:p-1)])))}
   if(is.null(upper)) {up=rep(Inf,length(parA))}
 if(is.vector(lower)){low=c(lower[-c(1)]*K,rep(0,length(parA[-c(1:p-1)])))}
   if(is.null(lower) & equal.ang==FALSE) {low=c(rep(-Inf,length(parA[c(1:p-1)])),rep(0,length(parA[-c(1:p-1)])))}
   if(is.null(lower) & equal.ang==TRUE) {low=c(rep(0,length(parA)))}

timing=system.time(res<-try(maxL.BFGS.B(if(equal.com==TRUE & equal.ang==FALSE)objective1max  else if(equal.com==FALSE & equal.ang==FALSE)objective2max else if(equal.com==TRUE & equal.ang==TRUE) objective3max else objective4max,grad=if(equal.com==TRUE & equal.ang==FALSE)objective1gr  else if(equal.com==FALSE & equal.ang==FALSE)objective2gr else if(equal.com==TRUE & equal.ang==TRUE) objective3gr else objective4gr,start=parA,up,low)))

if(is.character(res)==TRUE & try.refit.BFGS==TRUE){

cat("","\n");
cat("Fail to converge. Re-estimate removing default box constraints on z,v and a parameters... \n");
cat("","\n");

 if(is.vector(upper)) {up=c(upper[-c(1)]*K,rep(Inf,length(parA[-c(1:p-1)])))}
   if(is.null(upper)) {up=rep(Inf,length(parA))}
 if(is.vector(lower)){low=c(lower[-c(1)]*K,rep(-Inf,length(parA[-c(1:p-1)])))}
   if(is.null(lower) & equal.ang==FALSE) {low=c(rep(-Inf,length(parA)))}
   if(is.null(lower) & equal.ang==TRUE) {low=c(rep(-Inf,length(parA)))}



timing=system.time(res<-maxL.BFGS.B(if(equal.com==TRUE & equal.ang==FALSE)objective1max  else if(equal.com==FALSE & equal.ang==FALSE)objective2max else if(equal.com==TRUE & equal.ang==TRUE) objective3max else objective4max,grad=if(equal.com==TRUE & equal.ang==FALSE)objective1gr  else if(equal.com==FALSE & equal.ang==FALSE)objective2gr else if(equal.com==TRUE & equal.ang==TRUE) objective3gr else objective4gr,start=parA,up,low))
}

if(is.character(res)==TRUE){
cat("","\n");
cat("CONVERGENCE FAILURE: REDUCE m AND/OR lmm VALUES (default: lmm =",length(parA),"; suggested 3=<lmm<=20)... \n");
cat("","\n");
}

if(res$maximum>0){cat("Fail to converge, discrepancy function value is negative. Try to reduce m \n"); break}
if(print.level == 1) {
	  cat("\n")
      cat("Final gradient value:\n")
      print(res$gradient)
   }
param=res$est 
if(equal.ang==FALSE){res$est[1:(p-1)]=res$est[1:(p-1)]/K

for(i in 1:(p-1)){
if((res$est[i])<0)
res$est[i]=res$est[i]+360
else if((res$est[i])>360)
res$est[i]=res$est[i]-360
  }}

if(mcsc=="0"){
if(m==1){b=  c(1,1)} else {b=  c(res$est[((p-1)+1)],1,res$est[((p-1)+2):((p-1)+(m))]);b[1]=sum(b[seq(2,m+1,by=2)])-sum(b[seq(3,m+1,by=2)])}
res$est[((p-1)+1)]=b[1];param[((p-1)+1)]=b[1]
             }






hess=if(equal.com==TRUE & equal.ang==FALSE)objective1hess(param)  else if(equal.com==FALSE & equal.ang==FALSE)objective2hess(param) else if(equal.com==TRUE & equal.ang==TRUE) objective3hess(param) else objective4hess(param)

qr.hess <- try(if(mcsc=="unconstrained"){qr(hess)} else if(mcsc=="0"){qr(hess[-c(p),-c(p)])} else if(mcsc=="-1"){qr(hess[-c(p,seq(p+1,p-1+m,by=2)),-c(p,seq(p+1,p-1+m,by=2))])}, silent=TRUE)
        if (inherits(qr.hess, "try-error")){
            warning("\n\n\nCould not compute QR decomposition of Hessian.\nOptimization probably did not converge.\n*Increase the maximum number of iterations (iterlim) the computer will attempt in seeking convergence.\n*Increase the parameter factr.")
			}

solve.hess <- try(if(mcsc=="unconstrained"){(solve(hess))} else if(mcsc=="0"){(solve(hess[-c(p),-c(p)]))} else if(mcsc=="-1"){(solve(hess[-c(p,seq(p+1,p-1+m,by=2)),-c(p,seq(p+1,p-1+m,by=2))]))}, silent=TRUE)
        if (inherits(solve.hess, "try-error")){
   if(length(parA)>=1/2*(p*(p+1))){stop('\n\nSINGULAR HESSIAN: THE MODEL IS PROBABLY UNDERIDENTIFIED. \nTHE STANDARD ERRORS OF THE MODEL PARAMETER ESTIMATES CANNOT BE COMPUTED.\n\n')}
   else if(length(parA)<1/2*(p*(p+1))){stop('\n\nSINGULAR HESSIAN: THE MODEL IS PROBABLY MISSPECIFIED. \n*TRY TO REMOVE EQUALITY CONSTRAINTS (i.e., equal.ang=TRUE/equal.com=TRUE/mcsc="-1"/mcsc="0") AND/OR REDUCE m. \nTHE STANDARD ERRORS OF THE MODEL PARAMETER ESTIMATES CANNOT BE COMPUTED.\n\n')}
}


dhes=if(mcsc=="unconstrained"){diag(solve(hess))} else if(mcsc=="0"){diag(solve(hess[-c(p),-c(p)]))} else if(mcsc=="-1"){diag(solve(hess[-c(p,seq(p+1,p-1+m,by=2)),-c(p,seq(p+1,p-1+m,by=2))]))}
Sdev=sqrt(-2/N*dhes)
if(equal.ang==FALSE){Sdev[1:(p-1)]=Sdev[1:(p-1)]/K}
if(mcsc=="0"){
Sdev<-append(Sdev,0.000,p-1)
             }
if(mcsc=="-1"){
Sdev<-append(Sdev,0.000,p-1)
app<-seq(p,p-2+m,by=2)
for(i in 1:length(app))
Sdev<-append(Sdev,0.000,app[i])
             }

if(equal.ang==FALSE ){if(m==1){par.names=c(v.names,paste(rep("a",m),c(0)),if(equal.com==TRUE) rep("v",1) else  paste(rep("v",p),v.names),paste(rep("z",p),v.names))
}
 else
par.names=c(v.names,paste(rep("a",m),c(0,2:m)),if(equal.com==TRUE) rep("v",1) else  paste(rep("v",p),v.names),paste(rep("z",p),v.names));
coeff<-data.frame(c(0,round(res$est,5)),c(0,Sdev))}

if(equal.ang==TRUE ){if(m==1){par.names=c(v.names,paste(rep("a",m),c(0)),if(equal.com==TRUE) rep("v",1) else  paste(rep("v",p),v.names),paste(rep("z",p),v.names))
}
 else
par.names=c(v.names,paste(rep("a",m),c(0,2:m)),if(equal.com==TRUE) rep("v",1) else  paste(rep("v",p),v.names),paste(rep("z",p),v.names));
coeff<-data.frame(c(start.valuesA(R,k)$polar.angles,c(round(res$est,5))),c(rep(0,p),Sdev))}
names(coeff)<-c("Parameters","Stand. Errors")
row.names(coeff)<-par.names






 cilevel=(1-ci.level)/2
 pa.l<-round(qnorm(cilevel,coeff[1:p,1],coeff[1:p,2]))
 pa.u<-round(qnorm((1-cilevel),coeff[1:p,1],coeff[1:p,2]))
 for(i in 1:p){
 if((pa.l[i])>360)pa.l[i]=pa.l[i]-360
       if((pa.l[i])<0)pa.l[i]=pa.l[i]+360

   }
 for(i in 1:p){
 if((pa.u[i])>360)pa.u[i]=pa.u[i]-360
       if((pa.u[i])<0)pa.u[i]=pa.u[i]+360

   }
pestconfi=data.frame(round(coeff[1:p,1]),paste("(",pa.l),c(rep(";",p)),paste(pa.u,")"),round(360-coeff[1:p,1]),paste("(",360-pa.l),c(rep(";",p)),paste(360-pa.u,")"))
names(pestconfi)<-c("estimates","(L",";","U)","360-estimates","(L",";","U)")
row.names(pestconfi)<-v.names[1:p]
POSpa<-order(round(coeff[1:p,1]))
pestconfi<-pestconfi[POSpa,]

if(equal.com==TRUE) vii=coeff["v",1] else vii=coeff[paste(rep("v",p),v.names),1]
if(equal.com==TRUE) vii.s.e=coeff["v",2]  else vii.s.e=coeff[paste(rep("v",p),v.names),2]
vii.l=vii*exp(qnorm(cilevel)*vii.s.e/vii)
vii.u=vii/exp(qnorm(cilevel)*vii.s.e/vii)
cestconfi=data.frame(round(  rep(sqrt(1/(1+vii)),if(equal.com==TRUE)p else 1),2),paste("(",round(  rep(sqrt(1/(1+vii.u)), if(equal.com==TRUE)p else 1),2)),c(rep(";",p)),  paste(   round(  rep(sqrt(1/(1+vii.l)),if(equal.com==TRUE)p else 1),2),")"))
names(cestconfi)<-c("estimates","(L",";","U)")
row.names(cestconfi)<-v.names
cestconfi<-cestconfi[POSpa,]



conf.level=0.90
n=N-1
q=if(mcsc=="unconstrained"){length(parA)}else if(mcsc=="0"){length(parA)-1}else{length(parA)-(m-1)}
criterion=-1*c(res$max); chisq=criterion*n; d=1/2*(p*(p+1))-q;

if(length(res$Active.Bounds.names)!=0){d2=1/2*(p*(p+1))-q+length(res$Active.Bounds.names)}

RMSEA=round(sqrt(max(   (criterion*n-(p*(p+1)/2-q))/(n*(p*(p+1)/2-q))  ,0)),3)

if(length(res$Active.Bounds.names)!=0){RMSEA2=round(sqrt(max(   (criterion*n-(p*(p+1)/2-q+length(res$Active.Bounds.names)))/(n*(p*(p+1)/2-q+length(res$Active.Bounds.names)))  ,0)),3)}
tail=1/2*(1-conf.level); max=1000; 


 upper.rmsea = try(uniroot(function(l) ifelse(l>0,pchisq(ncp=l,q=chisq,df=d)-0.05,1),c(0,N*100))$root,silent=TRUE)
 lower.rmsea = try(uniroot(function(l) ifelse(l>0,pchisq(ncp=l,q=chisq,df=d)-0.95,1),c(0,N*100))$root,silent=TRUE)
 if(is.character(upper.rmsea)==TRUE){RMSEA.U=NA;warning("cannot find upper bound of RMSEA")}else  RMSEA.U <- round(sqrt(max((upper.rmsea)/(n*d),0)),3)
 if(is.character(lower.rmsea)==TRUE){RMSEA.L=NA;warning("cannot find lower bound of RMSEA")}else  RMSEA.L <- round(sqrt(max((lower.rmsea)/(n*d),0)),3)
if(length(res$Active.Bounds.names)!=0){
 upper.rmsea2 = try(uniroot(function(l) ifelse(l>0,pchisq(ncp=l,q=chisq,df=d2)-0.05,1),c(0,N*100))$root,silent=TRUE)
 lower.rmsea2 = try(uniroot(function(l) ifelse(l>0,pchisq(ncp=l,q=chisq,df=d2)-0.95,1),c(0,N*100))$root,silent=TRUE)
 if(is.character(lower.rmsea2)==TRUE){RMSEA.L2=NA;warning("cannot find lower bound of RMSEA")}else RMSEA.L2 <- round(sqrt(max((lower.rmsea2)/(n*d2),0)),3)
 if(is.character(upper.rmsea2)==TRUE){RMSEA.U2=NA;warning("cannot find upper bound of RMSEA")}else RMSEA.U2 <- round(sqrt(max((upper.rmsea2)/(n*d2),0)),3)

	
	}

Fzero=round(RMSEA^2*d,3)
Fzero.U=round(RMSEA.U^2*d,3)
Fzero.L=round(RMSEA.L^2*d,3)

if(length(res$Active.Bounds.names)!=0){

Fzero2=round(RMSEA2^2*d2,3)
Fzero.U2=round(RMSEA.U2^2*d2,3)
Fzero.L2=round(RMSEA.L2^2*d2,3)


}

Test.stat=criterion*n
Ho.perfect.fit=0.00^2*d*n
Ho.close.fit=0.05^2*d*n
Test.stat=round(Test.stat,2)
T1=round(pchisq(Test.stat,d,ncp=Ho.perfect.fit,lower.tail=FALSE),3)
T2=round(pchisq(Test.stat,d,ncp=Ho.close.fit,lower.tail=FALSE),3)


alpha.power=0.05;
Hi.strclfit=0.01^2*d*n
Hi.nclfit0.08=0.08^2*d*n
Hi.nclfit0.06=0.06^2*d*n
power.not.close.fit=round(pchisq(qchisq(alpha.power,d,ncp=0.05^2*n*d,lower.tail=TRUE),d,ncp=Hi.strclfit,lower.tail=TRUE),3)
power.close.fit1=round(pchisq(qchisq(alpha.power,d,ncp=0.05^2*n*d,lower.tail=FALSE),d,ncp=Hi.nclfit0.08,lower.tail=FALSE),3)
power.close.fit2=round(pchisq(qchisq(alpha.power,d,ncp=0.05^2*n*d,lower.tail=FALSE),d,ncp=Hi.nclfit0.06,lower.tail=FALSE),3)
power.exact.fit=round(pchisq(qchisq(alpha.power,d,ncp=0.00,lower.tail=FALSE),d,ncp=0.05^2*n*d,lower.tail=FALSE),3)



if(length(res$Active.Bounds.names)!=0){


Ho.perfect.fit2=0.00^2*d2*n
Ho.close.fit2=0.05^2*d2*n

T12=round(pchisq(Test.stat,d2,ncp=Ho.perfect.fit2,lower.tail=FALSE),3)
T22=round(pchisq(Test.stat,d2,ncp=Ho.close.fit2,lower.tail=FALSE),3)


alpha.power=0.05;
Hi.strclfit2=0.01^2*d2*n
Hi.nclfit0.082=0.08^2*d2*n
Hi.nclfit0.062=0.06^2*d2*n
power.not.close.fit2=round(pchisq(qchisq(alpha.power,d2,ncp=0.05^2*n*d2,lower.tail=TRUE),d2,ncp=Hi.strclfit2,lower.tail=TRUE),3)
power.close.fit12=round(pchisq(qchisq(alpha.power,d2,ncp=0.05^2*n*d2,lower.tail=FALSE),d2,ncp=Hi.nclfit0.082,lower.tail=FALSE),3)
power.close.fit22=round(pchisq(qchisq(alpha.power,d2,ncp=0.05^2*n*d2,lower.tail=FALSE),d2,ncp=Hi.nclfit0.062,lower.tail=FALSE),3)
power.exact.fit2=round(pchisq(qchisq(alpha.power,d2,ncp=0.00,lower.tail=FALSE),d2,ncp=0.05^2*n*d2,lower.tail=FALSE),3)



}


R=R;dimnames(R)=list(v.names,v.names)
S=if(equal.com==TRUE & equal.ang==FALSE)attr(objective1max(param),"S")  else if(equal.com==FALSE & equal.ang==FALSE)attr(objective2max(param),"S") else if(equal.com==TRUE & equal.ang==TRUE) attr(objective3max(param),"S") else attr(objective4max(param),"S");dimnames(S)=list(v.names,v.names)
Cs=if(equal.com==TRUE & equal.ang==FALSE)attr(objective1max(param),"Cs")  else if(equal.com==FALSE & equal.ang==FALSE)attr(objective2max(param),"Cs") else if(equal.com==TRUE & equal.ang==TRUE) attr(objective3max(param),"Cs") else attr(objective4max(param),"Cs");dimnames(Cs)=list(v.names,v.names)
Pc=if(equal.com==TRUE & equal.ang==FALSE)attr(objective1max(param),"Pc")  else if(equal.com==FALSE & equal.ang==FALSE)attr(objective2max(param),"Pc") else if(equal.com==TRUE & equal.ang==TRUE) attr(objective3max(param),"Pc") else attr(objective4max(param),"Pc");dimnames(Pc)=list(v.names,v.names)
I=diag(1,dim(S)[1])
SR=(solve(S)%*%R)%*%(solve(S)%*%R)
SRS=(solve(S)%*%R-I)%*%(solve(S)%*%R-I)

        CC <- diag(diag(R))
        chisqNull<- n*(sum(diag(R %*% solve(CC))) + log(det(CC)) -log(det(R)) - p)
        dfNull <- p*(p - 1)/2
        NFI <- round((chisqNull - chisq)/chisqNull,3)
        NNFI <- round((chisqNull/dfNull - chisq/d)/(chisqNull/dfNull - 1),3)
        
        if(length(res$Active.Bounds.names)!=0){NNFI2 <- round((chisqNull/dfNull - chisq/d2)/(chisqNull/dfNull - 1),3)}
        
        L1 <- max(chisq - d, 0)
        L0 <- max(L1, chisqNull - dfNull)
        CFI <- round(1 - L1/L0,3)
        
        if(length(res$Active.Bounds.names)!=0){
        L12 <- max(chisq - d2, 0)
        L02 <- max(L1, chisqNull - dfNull)
        CFI2 <- round(1 - L1/L0,3)
        	}
        
       residuals <- if(prod(diag(R))==1)R-S else R-Cs
       esse <- diag(R)
       standardized.residuals=residuals/sqrt(outer(esse, esse))
       SRMR <- round(sqrt(sum(standardized.residuals^2 * 
        upper.tri(diag(p), diag=TRUE))/(p*(p + 1)/2)),3)

GFI=round(p/(p+2*Fzero),3) 
      if(length(res$Active.Bounds.names)!=0){GFI2=round(p/(p+2*Fzero2),3)}
AGFI=round(1-((1/(2*d)*p*(p+1))*(1-GFI)),3) 
      if(length(res$Active.Bounds.names)!=0){AGFI2=round(1-((1/(2*d2)*p*(p+1))*(1-GFI2)),3)}
AIC=round(criterion-2*q/n,3)
CAIC=round(chisq-(log(N)+1)*q,3)
BIC=round(criterion-q*log(N)/n,3)
BCI=round((RMSEA^2*d+d/n)+2*q/(n-p-1),3)
      if(length(res$Active.Bounds.names)!=0){BCI2=round((RMSEA2^2*d2+d2/n)+2*q/(n-p-1),3)}
BCI.U=round((RMSEA.U^2*d+d/n)+2*q/(n-p-1),3)
BCI.L=round((RMSEA.L^2*d+d/n)+2*q/(n-p-1),3)
      if(length(res$Active.Bounds.names)!=0){
      	BCI.U2=round((RMSEA.U2^2*d2+d2/n)+2*q/(n-p-1),3)
          BCI.L2=round((RMSEA.L2^2*d2+d2/n)+2*q/(n-p-1),3)
      	}
CNI=((qnorm(0.95)+sqrt((2*d-1)))^2/(2*chisq/(N-1)))+1
      if(length(res$Active.Bounds.names)!=0){CNI2=((qnorm(0.95)+sqrt((2*d2-1)))^2/(2*chisq/(N-1)))+1}

	if(m==1){
if(equal.ang==FALSE)a.iter=param[(p)]
if(equal.ang==TRUE)a.iter=param[(1)]
   b.iter=c(a.iter[1]/(sum(a.iter)+1),1/(sum(a.iter)+1))
          } 
        else{
if(equal.ang==FALSE)a.iter=param[(p):(p+m-1)]
if(equal.ang==TRUE)a.iter=param[(1):(m)]
   b.iter=c(a.iter[1]/(sum(a.iter)+1),1/(sum(a.iter)+1),a.iter[-c(1)]/(sum(a.iter)+1))
          }


theta=K*c(0:360)
rho=rep(0,length(theta))
for(i in 1:length(rho)){
        
        rho[i]=b.iter[1]+c(rep(1,length(b.iter[-c(1)])))%*%(b.iter[-c(1)]*cos(c(seq(1,m))*theta[i] ))  
        
                            } 
mincorr180=2*(sum(b.iter[c(seq(1,(m+1),by=2))]))-1;summary(rho);
mincorr180=round(mincorr180,3)



 
cat("\n           =================================")
cat("\n              MEASURES OF FIT OF THE MODEL  ")
cat("\n           =================================","\n")

if(length(res$Active.Bounds.names)!=0){	
if(length(res$Active.Bounds.names)==1){cat("\n NOTE: ONE PARAMETER (",res$Active.Bounds.names,") IS ON A BOUNDARY.\n\n-----------Model degrees of freedom=",d,"\n           Active Bound=",length(res$Active.Bounds.names),"\n           The appropriate distribution for the test statistic lies between \n           chi-squared distribution with",d,"and with", d,"+",length(res$Active.Bounds.names),"degrees of freedom.\n\n-----------Values enclosed in square brackets are based on", d,"+",length(res$Active.Bounds.names),"=",d2,"degrees of freedom.\n")}
if(length(res$Active.Bounds.names)>1){cat("\n NOTE:",length(res$Active.Bounds.names)," PARAMETERS (",paste(res$Active.Bounds.names,";"),") ARE ON A BOUNDARY.\n\n-----------Model degrees of freedom=",d,"\n           Active Bounds=",length(res$Active.Bounds.names),"\n           The appropriate distribution for the test statistic lies between \n           chi-squared distribution with",d,"and with", d,"+",length(res$Active.Bounds.names),"degrees of freedom.\n-----------Values enclosed in square brackets are based on", d,"+",length(res$Active.Bounds.names),"=",d2,"degrees of freedom.\n")}
}


cat("\n-----------Sample discrepancy function value        :",round(criterion,3),"\n")
cat("\n-----------Population discrepancy function value, Fo ")
cat("\n           Point estimate                           :",Fzero ,if(length(res$Active.Bounds.names)!=0){paste("[",Fzero2,"]")})
cat("\n           Confidence Interval",conf.level*100,"%","                :","(",Fzero.L,if(length(res$Active.Bounds.names)!=0){paste("[",Fzero.L2,"]")},";",Fzero.U,if(length(res$Active.Bounds.names)!=0){paste("[",Fzero.U2,"]")},")","\n");


cat("\n-----------ROOT MEAN SQUARE ERROR OF APPROXIMATION ")
cat("\n           Steiger-Lind: RMSEA=sqrt(Fo/Df) ")
cat("\n           Point estimate                           :",RMSEA,if(length(res$Active.Bounds.names)!=0){paste("[",RMSEA2,"]")});                                                            cat("\n           Confidence Interval",conf.level*100,"%","                :","(",RMSEA.L,if(length(res$Active.Bounds.names)!=0){paste("[",RMSEA.L2,"]")},";",RMSEA.U,if(length(res$Active.Bounds.names)!=0){paste("[",RMSEA.U2,"]")},")","\n");


cat("\n-----------Discrepancy function TEST ")
cat("\n           TEST STATISTIC                           :",Test.stat)
cat("\n           p values:");                                                                    cat("\n           Ho: perfect fit (RMSEA=0.00)             :",T1,if(length(res$Active.Bounds.names)!=0){paste("[",T12,"]")});
cat("\n           Ho: close fit (RMSEA=0.050)              :",T2,if(length(res$Active.Bounds.names)!=0){paste("[",T22,"]")});
cat("\n");
cat("\n-----------Power estimation (alpha=0.05),");
cat("\n           N",N);
cat("\n           Degrees of freedom=",d,if(length(res$Active.Bounds.names)!=0){paste("[",d2,"]")});
cat("\n           Effective number of parameters=",q);
cat("\n       Ho (RMSEA=0.05) vs Alternative (RMSEA=0.01) :",power.not.close.fit,if(length(res$Active.Bounds.names)!=0){paste("[",power.not.close.fit2,"]")});
cat("\n       Ho (RMSEA=0.05) vs Alternative (RMSEA=0.06) :",power.close.fit2,if(length(res$Active.Bounds.names)!=0){paste("[",power.close.fit22,"]")});
cat("\n       Ho (RMSEA=0.05) vs Alternative (RMSEA=0.08) :",power.close.fit1,if(length(res$Active.Bounds.names)!=0){paste("[",power.close.fit12,"]")});
cat("\n       Ho (RMSEA=0.00) vs Alternative (RMSEA=0.05) :",power.exact.fit,if(length(res$Active.Bounds.names)!=0){paste("[",power.exact.fit2,"]")});

cat("\n")
cat("\n-----------EXPECTED CROSS VALIDATION INDEX ")
cat("\n           Browne and Cudeck's Index BCI-(MODIFIED AIC) ")
cat("\n           Point estimate                          :",BCI,if(length(res$Active.Bounds.names)!=0){paste("[",BCI2,"]")});                                                            cat("\n           Confidence Interval",conf.level*100,"%","               :","(",BCI.L,if(length(res$Active.Bounds.names)!=0){paste("[",BCI.L2,"]")},";",BCI.U,if(length(res$Active.Bounds.names)!=0){paste("[",BCI.U2,"]")},")","\n");
cat("\n           Hoelter's CN( .05 )                     :",round(CNI),if(length(res$Active.Bounds.names)!=0){paste("[",round(CNI2),"]")})
cat("\n")
cat("\n-----------Fit index")
cat("\n           Chisquare (null model) = ", chisqNull,  "  Df = ", dfNull)
cat("\n           Bentler-Bonnett NFI                     :",NFI)
cat("\n           Tucker-Lewis NNFI                       :",NNFI,if(length(res$Active.Bounds.names)!=0){paste("[",NNFI2,"]")})
cat("\n           Bentler CFI                             :",CFI,if(length(res$Active.Bounds.names)!=0){paste("[",CFI2,"]")})
cat("\n           SRMR                                    :",SRMR)
cat("\n           GFI                                     :",GFI,if(length(res$Active.Bounds.names)!=0){paste("[",GFI2,"]")})
cat("\n           AGFI                                    :",AGFI,if(length(res$Active.Bounds.names)!=0){paste("[",AGFI2,"]")})
cat("\n-----------Parsimony index")
cat("\n           Akaike Information Criterion            :",AIC)
cat("\n           Bozdogans's Consistent AIC              :",CAIC)
cat("\n           Schwarz's Bayesian Criterion            :",BIC)
cat("\n")
cat("\n----------------------------------------");
cat("\n Parameter estimates and Standard Errors "); 
cat("\n----------------------------------------","\n");

print(round(coeff,5))


if(length(res$Active.Bounds.names)!=0){
cat("\n NOTE! ACTIVE BOUNDS FOR: ",paste(res$Active.Bounds.names,";"),"\n"); 
}


cat("\n---------------------------------------------------------------------------");
cat("\n Estimates (ML) of POLAR ANGLES and COMMUNALITY INDICES"); 
cat("\n (approximate,",ci.level*100,"% one at time confidence intervals)"); 
cat("\n Note: variable names have been reordered to yield increasing polar angles"); 
cat("\n---------------------------------------------------------------------------","\n");

CONFIDENCE<-data.frame(pestconfi,cestconfi)
names(CONFIDENCE)<-c("ang. pos.","(L",";","U)","360-ang. pos.","(L",";","U)","comm. ind.","(L",";","U)")
print(CONFIDENCE)
cat("\n")

cat("\n (MCSC) Correlation at 180 degrees:",mincorr180,"\n");
cat("----------------------------------------------------\n")
B=matrix(round(b.iter,4),1,length(b.iter))
colnames(B)=paste(rep("b",(m+1)),c(0:m))
rownames(B)<-c("Estimates of Betas:")
print(B)
cat("----------------------------------------------------")
cat("\n CPU Time for optimization",timing[1],"sec.","(",round(timing[1]/60),"min.)\n");
cat( "                               \n")
cat( "                               \n")
sink()
result=list()
result$coeff=coeff
result$R=R
result$S=S
result$Cs=Cs
result$Pc=Pc
result$n=n
result$q=q
result$d=d
result$criterion=criterion
pestconfi=data.frame(round(coeff[1:p,1]),round(qnorm(cilevel,coeff[1:p,1],coeff[1:p,2])),round(qnorm(1-cilevel,coeff[1:p,1],coeff[1:p,2])))
names(pestconfi)<-c("estimates","(L;","U)")
row.names(pestconfi)<-v.names[1:p]
result$polar.angles=pestconfi
result$chisq=chisq
result$RMSEA=RMSEA
result$RMSEA.U=RMSEA.U
result$RMSEA.L=RMSEA.L
result$Fzero=Fzero
result$Fzero.U=Fzero.U
result$Fzero.L=Fzero.L
        result$chisqNull<- chisqNull
        result$dfNull <- dfNull
        result$NFI <- NFI
        result$NNFI <- NNFI
        result$CFI <- CFI
        result$residuals <- residuals
        result$standardized.residuals <- standardized.residuals
        result$SRMR <- SRMR 

result$GFI=GFI
result$AGFI=AGFI
result$CNI=CNI
result$BCI=BCI
result$AIC=AIC
result$CAIC=CAIC
result$BIC=BIC
result$MCSC=mincorr180
result$v.names=v.names
result$beta=b.iter
result$equal.com=equal.com
result$equal.ang=equal.ang
if(equal.com==TRUE) com=1/(1+coeff["v",1]) else com=1/(1+coeff[paste(rep("v",p),v.names),1])
result$communality=com
result$communality.index=sqrt(com)
result$m=m
result$upper=upper
result$lower=lower
   invisible(result)
   
    }

