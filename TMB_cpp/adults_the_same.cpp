// TMB model for CKMR "adults_the_same" example
//author: Paul Conn
//https://ckmr.netlify.app/examples/estimation/adults_the_same
#include <TMB.hpp>

template<class Type>
Type plogis(Type x){
  return 1.0 / (1.0 + exp(-x));
}

template<class Type>
Type dbinom_kern_log(Type n, Type x, Type p){
  Type p0 = p==0;
  return x*log(p+p0)+(n-x)*log(1.0-p);
}


template<class Type>
Type get_PO_prob(int bi,int bj,int di,int am, Type Nbj){
  int iam = (bj-bi)>=am;
  int ideath = di>bj;
  return iam*ideath*2.0/Nbj;
}

template<class Type>
Type get_HS_prob(int bi,int bj,Type Nbj,Type phi){
  return 2.0*pow(phi,bj-bi)/Nbj;
}


template<class Type>
Type objective_function<Type>::operator() ()
{
  using namespace Eigen;  
  using namespace density;
  
  // Data
  DATA_INTEGER( nyrs );  //number of years modeled
  DATA_INTEGER( t_start ); //first year of genetic sampling
  DATA_INTEGER( am );  //age of maturity
  DATA_MATRIX( n_match_HS_bibj ); //number of half-sibling matches where older sibling is born in year b_i, and the younger is born in year b_j
  DATA_MATRIX( n_comp_HS_bibj ); //number of half-sibling comparisons
  DATA_ARRAY( n_match_PO_bidibj );  //number of parent-offspring matches where parent is born in year b_i, dies in year d_i, and the offspring is born in year b_j
  DATA_ARRAY( n_comp_PO_bidibj );  
  
  PARAMETER(phi_logit);
  PARAMETER(N0_trans);
  PARAMETER(lambda_trans);
  
  Type phi = plogis(phi_logit);
  Type lambda = plogis(lambda_trans)*0.2 + 0.9;
  vector<Type> N(nyrs);
  N(0)=100.0+exp(N0_trans);
  for(int it=1;it<nyrs;it++){
    N(it)=N(it-1)*lambda;
  }
  
  //fill probability lookup tables
  array<Type> PO_table(nyrs,nyrs,nyrs); //dimensions are parent birth year, parent death year, offspring birth year
  matrix<Type> HS_table(nyrs,nyrs); //dimensions are ind i's birth year, ind j's birth year 
  for(int ibi=0;ibi<(nyrs-1);ibi++){
    for(int ibj=ibi+1; ibj<nyrs; ibj++){
      HS_table(ibi,ibj)=get_HS_prob(ibi,ibj,N(ibj),phi);
      for(int idi=std::max(ibi,t_start);idi<nyrs;idi++){
        PO_table(ibi,idi,ibj)=get_PO_prob(ibi,ibj,idi,am,N(ibj));
      }
    }
  }
    
  //likelihood
  Type logl = 0;
  for(int ibi=0;ibi<(nyrs-1);ibi++){
    for(int ibj=(ibi+1);ibj<nyrs;ibj++){
      logl = logl + dbinom_kern_log(n_comp_HS_bibj(ibi,ibj),n_match_HS_bibj(ibi,ibj),HS_table(ibi,ibj)); //HSPs
      for(int idi = std::max(ibi,t_start); idi<nyrs; idi++){
        logl = logl + dbinom_kern_log(n_comp_PO_bidibj(ibi,idi,ibj),n_match_PO_bidibj(ibi,idi,ibj),PO_table(ibi,idi,ibj)); //POPs
      }
    }
  }
  
  //to get access to most recent values computed from arguments passed to the function
  REPORT(PO_table);
  REPORT(HS_table);
  REPORT(N);
  REPORT(phi);
  REPORT(lambda)
    
  //things you want standard errors of    
  ADREPORT( N );
  ADREPORT( phi );
  ADREPORT( lambda );

  return -logl;
}
