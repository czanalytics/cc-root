/*  
 * Simulate electron e e -> mu mu  -scattering experiment
 * using ROOT, http://en.wikipedia.org/wiki/ROOT
 *
 * a) hit-and-miss (von Neuman rejection) MC method
 * b) Maximum likelihood fit for function of 2 parameters
 * c) use of c++11 lambda function and functor
 *  
 * f(x, alfa, beta) = (1 + alfa*x + beta*x*x) / (2 + 2*beta/3)
 *
 * For limited range [xmin, xmax], the normalized function is
 * f(x, alfa, beta) = (1 + alfa * x + beta * x*x) /
 * (xmax - xmin) + 0.5*alfa * (xmax^2 - xmin^2) + beta/3 * (xmax^3-xmin^3)
 *
 * Ref. Institute of Particle and Nuclear Physics
 * http://www-ucjf.troja.mff.cuni.cz/new/wp-content/uploads/2014/02/lesson3_example03.cpp
 *
 */

#include "TH1D.h"
#include "TH2D.h"
#include "TRandom3.h"
#include "TCanvas.h"
#include <math.h>
#include "TStyle.h"
#include <functional>
#include "TFile.h"
#include "TTree.h"

#include "Minuit2/Minuit2Minimizer.h"
#include "Math/Functor.h"

#include <iostream>
#include <vector>
#include <iomanip>

using std::cout;
using std::endl;
using std::vector;

const double xmin = -0.95;
const double xmax =  0.95;
const double normal  = (xmax - xmin);
const double normal2 = 0.5 * (xmax*xmax - xmin*xmin);
const double normal3 = 1/3. * (xmax*xmax*xmax - xmin*xmin*xmin);

double f_scatter( double x, double alfa, double beta) {
   // normalization unimportant for hit and miss, usefull for likelihoods
   return ( 1 + alfa*x + beta* x*x) / 
      ( normal + alfa*normal2 + beta*normal3 ); 
}

// initialize - finds maximum of the function
double initialize_generator(double &fmax,  std::function<double (double)> function ) {
   fmax = function(xmin);
   const double step = 0.01;
   double x = xmin;
   while( x <= xmax ) {
      x += step;
      double f = function(xmin);
      if(f>fmax) fmax = f;
   }
}

TRandom3 rnd(1236);

// MC event
double generate_event(double & fmax, std::function<double (double)> function ) {
   int n = 0;
   double x, y, f;
   while(1){
      x =rnd.Uniform(xmin, xmax);
      f = function(x);
      if(f> fmax) fmax = f; // new maximum found
      y = rnd.Uniform(0, fmax);
      if(y < f) return x;
      n++;
      if(n>1e8){
         cout << "Generation not efficient > 1e8 \n";
         exit(1);
      }
   }
}

int main(){
   gStyle-> SetOptStat(1111);

   // specific choice of true paramters for pseudo data
   double alfa = 0.5;
   double beta = 0.5;

   // wrap the multi parameter function to single parameter function
   auto f_truth = [=](double x) { return f_scatter(x, alfa, beta); }; // lambda
   // [=] is equivalent of [alfa,beta] 
   // [&] is equivalent of [&alfa,&beta] 

   // initialize generator
   double fmax;
   initialize_generator(fmax, f_truth);

   // create pseudo data
   int n = 1000;
   vector<double> events;
   for( size_t i=0; i<n; i++){
      events.push_back( generate_event(fmax, f_truth) );
   }

   // analysis and dump info to tree
   TH1D *hdata = new TH1D("hdata", "hdata; cos(#theta); Events", 40, -1, 1);

   TFile * fout = TFile::Open("scatter.root", "recreate");
   TTree *tree = new TTree("scatter", "Sample of scattering");

   Int_t split = 0;
   Int_t bsize = 64000;

   float x;
   tree->Branch("x", &x, "x/F");

   for( size_t i=0; i<n; i++) {
      hdata-> Fill( events[i] );
      // to save into tree -> tree will be used with RooFit
      x = events[i];
      tree-> Fill();
   }

   fout-> Write();
   fout-> Close();

   TCanvas *cf = new TCanvas("cf", "cf", 0, 0, 800, 600);
   hdata-> Draw();
   cf-> SaveAs("data_scatter.eps");

   // ML analysis

   // two parameter (xx[0], xx[1]) likelihood function which takes data (events) as input
   auto logLike = [&events](const double *xx) { // lambda
      double alfa = xx[0];
      double beta = xx[1];
      double logL = 0;

      for( size_t i=0; i<events.size(); i++) {
         const double & x = events[i];
         logL += log(f_scatter(x, alfa, beta));
      }
      return -logL;
   };

   // Now set the infrastructure to minimize maximize LogL (or minimize -logL)
   // adapted from: http://root.cern.ch/drupal/content/numerical-minimization

   // Choose method upon creation between: kMigrad, kSimplex etc.
   ROOT::Minuit2::Minuit2Minimizer min ( ROOT::Minuit2::kMigrad );
 
   min.SetMaxFunctionCalls(1000000);
   min.SetMaxIterations(100000);
   min.SetTolerance(0.001);
 
   // wrapper of a function to Functor on which Minuit2Minimizer operaters
   ROOT::Math::Functor f(logLike, 2); // lambda

   // initial values of parameters and step size
   double step[2] = {0.01, 0.01};
   double variable[2] = {0.8, 0.8};
 
   min.SetFunction(f); // assign functor to minimizer
 
   // Set the free variables to be minimized!
   min.SetVariable(0, "x", variable[0], step[0]);
   min.SetVariable(1, "y", variable[1], step[1]);
 
   min.Minimize(); 
 
   // results
   const double *xs = min.X();
   cout << "Minimum: f( alfa = " << xs[0] << ", beta = " << xs[1] << "): " 
        << logLike(xs) << endl;
 
   double cov[4];
   min.GetCovMatrix(cov);

   cout << "Covariance matrix: \n"
   << cov[0] << std::setw(10) << cov[1] << "\n"
   << cov[2] << std::setw(10) << cov[3] << endl;

   cout << "Uncertainties: \n "
      << " sigma_alfa = " << sqrt(cov[0]) 
      << " sigma_beta = " << sqrt(cov[3])  
      << endl;
   cout << "Correlation rho =  " <<  cov[1] / sqrt(cov[0]) / sqrt(cov[3]) << endl;

   // For nexp similar experiments maximize likelihood
   // Estimate of the the statistical uncertainty from single experiment 
   // compares to statistica fluctuation from pseudoexperiments

   cout << "Calculating pseudoexperiments ... " << endl;

   int nexp = 500;

   TH2D *h_alfa_beta = new TH2D("h_alfa_beta", "h_alfa_beta;#hat{#alpha};#hat{#beta}", 40, 0, 1, 40, 0, 1);
   TH1D *h_alfa = new TH1D("h_alfa", "h_alfa;#hat{#alpha};events", 40, 0, 1);
   TH1D *h_beta = new TH1D("h_beta", "h_beta;#hat{#beta};events", 40, 0, 1);
   TH1D *h_logLmax = new TH1D("h_logLmax", "Minimum of -logLikelihood;Min(-logLikelihood);events", 5000, 550, 650);

   for( size_t ie = 0; ie<nexp ; ie++) {
      events.clear();
      for( size_t i=0; i<n; i++){
         events.push_back( generate_event(fmax, f_truth) );
      }

      double variable[2] = {0.8, 0.8};
      // Set the free variables to be minimized!
      min.SetVariable(0, "x", variable[0], step[0]);
      min.SetVariable(1, "y", variable[1], step[1]);

      min.Minimize(); 

      const double *xs = min.X();

      h_alfa_beta-> Fill(xs[0], xs[1]);
      h_alfa-> Fill(xs[0]);
      h_beta-> Fill(xs[1]);
      h_logLmax -> Fill( logLike(xs)  );
   }

   cout << "Unceratinties from results of pseudoexperiments:"
      << " sigma_alfa = " <<  h_alfa-> GetRMS()
      << " sigma_beta = " <<  h_beta-> GetRMS()
   << endl;
   cout << "Correlation rho =  " <<  h_alfa_beta-> GetCorrelationFactor() << endl;

   cf-> Clear(); cf-> Divide(2,2);

   cf-> cd(1); h_alfa_beta-> Draw();
   cf-> cd(2); h_alfa-> Draw();
   cf-> cd(3); h_beta-> Draw();
   cf-> cd(4); h_logLmax-> Draw();

   cf-> SaveAs("mc_ml.pdf");
   return 0;
}

