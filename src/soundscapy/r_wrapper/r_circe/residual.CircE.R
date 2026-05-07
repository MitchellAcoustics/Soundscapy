residual.CircE<-function(object,file=NULL,digits=3){

if(!is.null(file))  sink(file,append=FALSE,split=TRUE)

  	coeff=object$coeff
	p=dim(object$R)[1]
	v.names=object$v.names
	R=round(object$R,digits=digits)
	S=object$S
     Cs=object$Cs
     Pc=object$Pc
     residuals=object$residuals
     stand.res=object$standardized.residuals
cat("\n","\n")
if(prod(diag(R))==1)cat("\n Sample Correlation Matrix  ","\n")else cat("\n Sample Covariance Matrix  ","\n")
print(R)
cat("\n .......................................................","\n")

cat("\n ","\n")
if(prod(diag(R))==1)cat("\n Reproduced Correlation Matrix  ","\n")else cat("\n Reproduced Covariance Matrix  ","\n")
if(prod(diag(R))==1)print(round(S,digits=digits)) else print(round(Cs,digits=digits))
cat("\n .......................................................","\n")

cat("\n ","\n")
cat("\n Reproduced Common Score Correlation Matrix  ","\n")
print(round(Pc,digits=digits))
cat("\n .......................................................","\n")

	   
cat("\n ","\n")
cat("\n Ratios of Reproduced Variances to Input Variances  ","\n")
ratio=round(diag(Cs)/diag(R),digits=digits)
print(ratio)
cat("\n .......................................................","\n")


cat("\n ","\n")
cat("\n Residual Matrix",if(prod(diag(R))==1)"(CORRELATION)"else"(COVARIANCE)","\n")
print(round(residuals,digits=digits))
cat("\n Residuals\n")
print(round(summary(as.vector(residuals)),digits=digits))
cat("\n Standardized Residuals\n")
print(round(summary(as.vector(stand.res)),digits=digits))
cat("\n .......................................................","\n")


sink()	

	}