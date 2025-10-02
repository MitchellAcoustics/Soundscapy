#' randall function
#'
#' Randomization test of hypothesized order relations
#'
#' @param df_list list of dataframes to be run, dataframes input should have only columns for the variables included in the models (e.g., 6 columns for RIASEC)
#' @param description any information you wish to enter describing each sample
#' @param ord prediction ordering (default "circular6").
#'     For circular models there are two preset inputs for 6 and 8 variables, "circular6" and "circular8".
#'     Also accepts a vector of prediction ordering, for example circular6 = c(1,2,3,2,1,1,2,3,2,1,2,3,1,2,1)
#'
#'
#' @return Data frame of RTHOR model results, one row per matrix
#'
#' @examples
#' randall_output <- RTHORR::randall_from_df(df_list = list(data1, data2),
#'                                           ord = "circular6",
#'                                           description = c("sample_one", "sample_two", "sample_three"))
#'
#' @import permute
#' @import gdata
#'
#' @export




my_randall_from_df<-function(df_list, description, ord = "circular8"){

  #make this its own function?
  #check that the input dataframes all have the same number of columns
  column_count <- vector(length = 0)
  for (d in df_list){
    column_count <- append(column_count, ncol(d))
  }
  if (length(unique(column_count))!=1){
    stop("Number of columns is not equal for all dataframes passed in.")
  }

  #check that number of descriptions is the same as the number of data frames passed in
  if (length(df_list) != length(description)){
    stop("Number of descriptions not equal to number of dataframes.")
  }



  #get number of matrixes to analyze
  nmat <- length(df_list)

  #get number of columns
  n <- ncol(df_list[[1]])


  #process ord input
  if (ord == "circular6"){
    ord = c(1,2,3,2,1,1,2,3,2,1,2,3,1,2,1)
  } else if (ord == "circular8"){
    ord=c(1,2,3,4,3,2,1,1,2,3,4,3,2,1,2,3,4,3,1,2,3,4,1,2,3,1,2,1)
  } else {
    ord = ord
  }


  # library("permute")
  setMaxperm<-(50000)
  requireNamespace("utils")
  np<- ((n*n)-n)/2


  #read input file
  #read in diagonal matrix as vector and then fill in
  # za<-scan(input)  #goodf read in vector

  #reverse order of df_list
  df_list <- rev(df_list)


  #empty vector for correlations
  za <- vector(length = 0)
  for (d in df_list){
    #run correlations and turn into vector
    cor_df <- cor(d)
    za <- append(gdata::lowerTriangle(cor_df, diag = TRUE, byrow = TRUE), za)
  }



  dmatm<-array(dim=c(n,n,nmat))
  for (m in 1:nmat){
    ii<-(m-1)*(np+n)

    for(j in 1:n){
      for(i in 1:n){
        if (i > j) next
        ii<-ii+1
        dmatm[i,j,m]<-za[ii]}}  #set up matrix


    ii<-(m-1)*(np+n)
    for(i in 1:n){
      for(j in 1:n){
        if (i  < j) next
        ii<-ii+1
        dmatm[i,j,m]<-za[ii]}}  #fill in
  } #end nmat loop


  #mathhyp generation
  mathyp<-matrix(nrow=np,ncol=np)
  for(i in 1:np){
    for(j in 1:np){
      if(ord[j]<ord[i]) mathyp[i,j]<-1 else mathyp[i,j]<-0}}


  #count hyp
  nhyp<-0
  for(i in mathyp){
    if(i==1)nhyp<-nhyp+1}

  #print header and nhyp   ********************************
  #out1<-matrix(nrow=2,ncol=2)

  zz<-matrix(nrow=1,ncol=n)
  out<- matrix(nrow=nmat,ncol=7)
  dmatp<-matrix(nrow=n,ncol=n)
  dmat<-matrix(nrow=n,ncol=n)

  pp<-vector(length=n)

  #set up permutation file
  nper<-1
  for(i in 1:n){
    nper<- i*nper}  ##good n permutation

  if(nper> 50000)nper<-50000
  permat<-matrix(nrow=nper,ncol=n)

  zz<-matrix(nrow=1,ncol=n)
  f<-function (zz){
    zz<-sample.int(n,n,replace=FALSE,prob=NULL)
    return(zz)}

  if(nper<50000)permat<-permute::allPerms(n, control = permute::how(maxperm = 50000), check = TRUE)

  if(nper>=50000) for(ll in 1:nper){permat[ll,]<-t(apply(zz,1,f))}



  #do big loop over nmat  kk
  for(kk in 1:nmat){

    #select matrix from array

    dmat<-dmatm[,,kk]

    #run on original data prior to permutations
    #do data 1,0 matc gt=1 eq =2


    nagr<-0
    ntie<-0

    ii=0
    scal<-vector(length=np)
    for(i in 1:n){
      for(j in 1:n){
        if (i >= j) next
        ii<-ii+1
        scal[ii]<-dmat[i,j]}}

    #match data matc with mathyp  1=conf 2=tie, 3=less

    matc <-matrix(nrow=np,ncol=np)
    for(i in 1:np){
      for(j in 1:np){
        if(scal[j] > scal[i]) matc[i,j]<-1
        if(scal[j] == scal[i]) matc[i,j]<-2
        if(scal[j] < scal[i]) matc[i,j]<-0}}
    nsup<-0
    nntie<-0
    for(i in 1:np){
      for(j in 1:np){
        if(matc[i,j]==1 & mathyp[i,j]==1) nagr<-nagr+1
        if(matc[i,j]==2 & mathyp[i,j]==1) ntie<-ntie+1}}

    ci<- (nagr-(nhyp-(nagr+ntie)))/nhyp


    count<- 1 #counter for number exceed values in original permuation

    nperx<-nper-1  ##check on this correction

    #do small loop over nper
    for(k in 1:nperx){   #minus 1
      #  if(k >50000) break  #stop if greater than 50,000

      #set up new matrix
      pp<-permat[k,]

      for(i in 1:n){
        for(j in 1:n){
          dmatp[i,j]<-dmat[pp[i],pp[j]]
        }}

      #do data 1,0 matc gt=1 eq =2

      ii=0
      scal2<-vector(length=np)
      for(i in 1:n){
        for(j in 1:n){
          if (i >= j) next
          ii<-ii+1
          scal2[ii]<-dmatp[i,j]}}

      ################################

      #match data matc with mathyp  1=conf 2=tie, 3=less

      # Replace first pair of nested for loops with vectorized operations
      matc2 <- matrix(nrow = np, ncol = np)
      matc2[] <- as.numeric(scal2[rep(1:np, each = np)] > scal2[rep(1:np, np)])
      matc2[scal2[rep(1:np, each = np)] == scal2[rep(1:np, np)]] <- 2

      # Replace second pair of nested for loops with vectorized operations
      nsup <- sum(matc2 == 1 & mathyp == 1)
      nntie <- sum(matc2 == 2 & mathyp == 1)

      ##################

      if(nsup >= nagr) count <- count+1   #count number of cases where fit is equal or greater


    } #end first loop kz
    prob<-count/nper

    out[kk,1]<-kk
    out[kk,2]<-nhyp
    out[kk,3]<-nagr
    out[kk,4]<-ntie
    out[kk,5]<-ci
    out[kk,6]<-prob
    # out[kk,7]<-samp[kk]
    out[kk,7]<-description[kk]


  }#end loop kk different matrices

  colnames(out)<-c("mat","pred","met","tie","CI","p","description")
  rownames(out)<-c(1:nmat)
  # print(out,quote=FALSE)


  out <- data.frame(out)
  out[,1:6] <- sapply(out[,1:6],as.numeric)

  # out %>% mutate_at('matrixnumber', as.numeric)

  return(out)
}  #end randall


reorder_mat <- function(mat, order){
  # Author: https://rdrr.io/cran/graph4lg/src/R/reorder_mat.R
  # Number of elements in the vector 'order'
  n <- length(order)

  # Check whether 'mat' is a 'matrix'
  if(!inherits(mat, "matrix")){
    stop("'mat' must be a matrix")
    # Check whether 'order' is of class 'character'
  } else if (!inherits(order, "character")){
    stop("'order' must be a character vector")
    # Check whether 'mat' is a symmetric matrix
  } else if(!(isSymmetric(mat))){
    stop("The matrix 'mat' must be symmetric")
    # Check whether 'order' has as many elements as there are rows
    # and columns in 'mat'
  } else if (n != length(colnames(mat))){
    stop("'order' must have as many elements as there are rows and
         columns in 'mat'")
    # Check whether the column names are in the 'order' vector
  } else if(length(which(colnames(mat) %in% order)) != n){
    stop("The column names of the matrix you want to reorder must
         be present in the vector 'order'")
    # Check whether the row names are in the 'order' vector
  } else if (length(which(row.names(mat) %in% order)) != n){
    print("The row names of the matrix you want to reorder must
          be present in the vector 'order'")
  } else {

    # Reorder 'mat' according to 'order'
    mat2 <- mat[order, order]

    return(mat2)
  }
}
