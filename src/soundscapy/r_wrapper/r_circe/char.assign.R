char.assign<-function(sc.names=NULL,v.names=NULL,point.char=NULL,bg.point=NULL){
	if(is.null(sc.names))stop("Argument sc.names is null and there is not default value. Insert the names of the main variables to search in v.names.")
	if(is.null(v.names))stop("Argument v.names is null and there is not default value. Insert the character vector where matches in given sc.names are sought.")
	p<-length(sc.names)
if(is.null(point.char))stop("Argument point.char is null and there is not default value. Give a vector of ",p, " different point character (see ?points)")
if(is.null(point.char))stop("Argument bg.points is null and there is not default value. Give a vector of ",p, " background (fill) color for the open plot symbols (see ?points and ?colors)")

	pchar<-rep(0,p)
	bg.points<-rep(0,p)
	for(i in 1:length(sc.names)){
		
		pos<-grep(sc.names[i],v.names)
		pchar[pos]<-point.char[i]
		bg.points[pos]<-bg.point[i]
		}
	result<-list()
	result$pchar<-pchar
	result$bg.points<-bg.points
	invisible(result)
	}