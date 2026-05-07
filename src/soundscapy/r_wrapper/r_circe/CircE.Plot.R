`CircE.Plot` <-
function(object,pchar=NULL,bg.points="red",ef=0.4,big.points=10,big.labels=10,bg.plot="gray80",col.axis="black",color="black",col.text="white",twodim=TRUE,bound=TRUE,labels=TRUE,reverse=FALSE){
	
	

dev.new(title="  FOURIER FUNCTION ",width=4,height=4)

k = object$m
K = pi/180

equal.ang=object$equal.ang
equal.com=object$equal.com
#------------------  Plot of correlation function  ----------------
b.iter=c(object$b)
#beta=res$parres$parres$par[(p+1):(p+(k+1))]
theta=K*c(0:360)
rho=rep(0,length(theta))
for(i in 1:length(rho)){
	
	rho[i]=b.iter[1]+c(rep(1,length(b.iter[-c(1)])))%*%(b.iter[-c(1)]*cos(c(seq(1,k))*theta[i] ))  
	
	                    } 
#plot(theta,rho,xlim=c(0*K,360*K),ylim=c(0,1),bty="n")
mincorr180=2*(sum(b.iter[c(seq(1,(k+1),by=2))]))-1;summary(rho);
mincorr180=round(mincorr180,3)


grad=c(0,45,90,135,180,225,270,315,360)
grad*K
pos=grad*K
par(plt=c(0.9,0.9,0.9,0.9),mar=c(3,3,2,1),cex.axis=0.7,pty="m",mgp=c(1.8,0.6,0),bg = bg.plot)
plot(theta,rho,xlim=c(0*K,360*K),ylim=if(mincorr180<0) c(-1,1) else c(0,1),bty="n",type="n",col=color,lwd=2,axes=FALSE,xlab=expression(Theta[d]),ylab=expression(rho),col.lab=col.axis,cex.lab=1.5)
#abline(v=pos[5],col="black",lty="dashed")
axis(1,pos,labels=grad,col.axis=col.axis,col=col.axis,las=1)
axis(2,col.axis=col.axis,col=col.axis,las=1)
#text(180*K,1,expression(rho(theta[d])==beta[o]+Sigma[k==1]^"m"*beta[k]*cos(k*theta[d])),col="black",cex=1.2,font=3)
title(main=expression(rho(theta[d])==beta[o]+Sigma[k==1]^"m"*beta[k]*cos(k*theta[d])),col.main=col.text,font.main=3)
text(182*K,0.966,substitute(list(rho[180*degree])==list(r),list(r=mincorr180)),col=color,cex=1.2)
for(i in 1:(k+1))
text(rep(0.825,length(b.iter)),if(mincorr180<0)c(-1+0.152*i)else c(0+0.152/2*i),labels=substitute(list(beta)[list(i)]==list(b.iter),list(b.iter=round(b.iter[i],4),i=(i-1))),cex=1.1,col=col.text)

points(theta,rho,type="l",col=color,lwd=2)




dev.new(title="  CIRCULAR POSITION  ",width=6,height=6)


v.names=object$v.names
#--------------------------- circular plot ----------------------
par(mar=c(0,0,0,0),bg = bg.plot)
if(equal.com==TRUE) com.ind=sqrt(1/(1+object$coeff["v",1])) else com.ind=sqrt(1/(1+object$coeff[paste("v",v.names),1]))
plot(c(-1:1),c(-1:1),type="n",bty="n",axes=FALSE);
x1=seq(0,max(com.ind),by=0.00001);x2=seq(-max(com.ind),0,by=0.001);x=c(x2,x1)
points(c(x,x),c(sqrt(max(com.ind)^2-x^2),c(-1)*sqrt(max(com.ind)^2-x^2)),cex=0.1,col=color,type="l",lwd=2,lty="solid");if(twodim==TRUE){abline(h=0,v=0,col=color,lty="solid",lwd=2)};
if(bound==TRUE){
if(!is.null(object$upper)){
up<-unique(object$upper)
segments(rep(0,length(up)),rep(0,length(up)),cos(up*K)*max(com.ind),sin(up*K)*max(com.ind))
}
if(!is.null(object$lower)){
low<-unique(object$lower)
segments(rep(0,length(low)),rep(0,length(low)),cos(low*K)*max(com.ind),sin(low*K)*max(com.ind))
}
}
angular.points=matrix(0,dim(object$R)[1],2)
for(i in 1:dim(object$R)[1]){
   if(reverse==FALSE){angular.points[i,]=c(cos(K*object$coeff[i,1]),sin(K*object$coeff[i,1]))}
   if(reverse==TRUE){angular.points[i,]=c(cos(K*(360-object$coeff[i,1])),sin(K*(360-object$coeff[i,1])))}   
             }
row.names(angular.points)=object$v.names
if(labels==TRUE){
text(angular.points[,1]*(max(com.ind)+ef*com.ind),angular.points[,2]*(max(com.ind)+ef*com.ind),col=color,pch=17,cex=(big.labels)/nrow(object$R),labels=v.names)}
segments(c(rep(0,dim(object$R)[1])),c(rep(0,dim(object$R)[1])),angular.points[,1]*max(com.ind),angular.points[,2]*max(com.ind),col=color,lty="dotted")
if(labels==TRUE){
segments(angular.points[,1]*max(com.ind),angular.points[,2]*max(com.ind),angular.points[,1]*(max(com.ind)+ef*com.ind),angular.points[,2]*(max(com.ind)+ef*com.ind),col="gray",lty="dotted")}
points(angular.points[,1]*com.ind,angular.points[,2]*com.ind,col="black",pch=if(is.null(pchar))21 else pchar,bg=bg.points,cex=(big.points)/nrow(object$R))
text(-1.09,0.92,labels=substitute(list(rho[180])==list(r),list(r=mincorr180)),col=col.text,pos=4);
text(-1.09,0.86,labels=substitute(list("max "*h)==list(maxcom),list(maxcom=round(max(com.ind),2))),col=col.text,pos=4)
text(-1.09,0.80,labels=substitute(list("max "*h^2)==list(maxcom),list(maxcom=round(max(com.ind^2),2))),col=col.text,pos=4)


}

