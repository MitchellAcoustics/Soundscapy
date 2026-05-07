bound.assign<-function(sc.names=NULL,v.names=NULL,lower=NULL,upper=NULL){
	if(is.null(sc.names))stop("Argument sc.names is null and there is not default value. Insert the names of the main variables to search in v.names.")
	if(is.null(v.names))stop("Argument v.names is null and there is not default value. Insert the character vector where matches in given sc.names are sought.")
	p<-length(sc.names)
if(is.null(lower))stop("Argument lower is null and there is not default value. Give a vector of ",p, " different lower limits")
if(is.null(upper))stop("Argument upper is null and there is not default value. Give a vector of ",p, " different upper limits")

	low<-rep(0,p)
	up<-rep(0,p)
	for(i in 1:length(sc.names)){
		
		pos<-grep(sc.names[i],v.names)
		low[pos]<-lower[i]
		up[pos]<-upper[i]
		}
	result<-list()
	result$upper<-up
	result$lower<-low
	invisible(result)
	}