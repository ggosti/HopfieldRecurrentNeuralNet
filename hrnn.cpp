
// Versione con ricerca sulle mappe (solo per mstati_di_mat, non per statim) e PARALLELIZZAZIONE BUONA

#include <pybind11/pybind11.h>
#include <pybind11/stl_bind.h>
#include <pybind11/numpy.h>

#include <cstdio>
#include <cstdlib>
#include <vector>
#include <array>
#include <ctime>
#include <map>
//#include <unordered_map>
#include <algorithm>
#include <iterator>
#include <fstream>
#include <iostream>
#include <iomanip>
#include <thread>
#include <future>
#include <mutex>
#include <assert.h>

#include <cmath>
#include <chrono>
#include <random>

#define HASH_MULT  314159    /* random multiplier */
#define HASH_PRIME 516595003 /* the 27182818th prime; it's $\leq 2^{29}$ */
#define N 14//5//14


using namespace std;

namespace py = pybind11;

using mappa = map<int, unsigned long long int>;

int show = 0;

// Definisco le funzioni dichiarate


// # Funzione per creare la matrice
void matrice(double* mat, double eps, double dil) { 
    double s_mat[N*N];
    double a_mat[N*N];
    unsigned long seed = 9999;//chrono::system_clock::now().time_since_epoch().count();
    default_random_engine re (seed);
    uniform_real_distribution<double> unif(0,1);
    double a1, a2; 

    for (int i = 0; i < N; i++) {
        for (int j = 0; j < N; j++) {
            mat[i+j*N] = 0;
            a_mat[i+j*N] = 0;
            s_mat[i+j*N] = 0;
        }
    }

    for (int i = 0; i < N; i++) {
        for (int j = i + 1; j < N; j++) {
            if (unif(re) > dil){
              a1 = (unif(re) - 0.5) * 2;                         // Gli elementi sono diversi da 0 solo con prob (1-dil)
              s_mat[i+j*N] = a1;                                  // Creo le due matrici, una simmetrica e una asimmetrica
              s_mat[j+i*N] = a1;                      
            }
            if (unif(re) > dil) {
              a2 = (unif(re) - 0.5) * 2;
              a_mat[i+j*N] = a2;
              a_mat[j+i*N] = -a2;
            }
        }
    }

    for (int i = 0; i < N; i++) {
        for (int j = 0; j < N; j++){
            mat[i+j*N] = (1 - eps / 2) * s_mat[i+j*N] + (eps / 2) * a_mat[i+j*N];   // La diagonale viene nulla per costruzione
        }
    }

}

// # Calcolo il prodotto tra matrice e vettore
void dot(double* mat, unsigned int* in, double* out) {
  double o;            
  for (int i = 0; i < N; ++i) {
    //py::list line;
    o=0; 
    for (int j = 0; j < N; ++j) {
        o += mat[i+N*j] * (double) in[j];
        //line.append(mat[i+j*N]);
        //line.append(in[j]);
    }
    //line.append(o);
    out[i]=o;
    //py::print(i,line);
  }
}


// # Creo lo stato "n"
void s_da_n(unsigned int* s0, int n) {
    for (int i = 0; i < N; i++) {
        s0[N-i-1] = (int(double(n) / (1 << i))) % 2;           // 1<<i è 2**i, calcolato molto velocemente
    }
}

// # Calcolo "n" corrispondente allo stato "s"
int n_da_s(unsigned int* s) {
    int ret = 0;

    for (int i = 0; i < N; i++){
        ret += s[i] * (1 << (N - 1 - i));               // Ovvero +=s[i]*2**(N-1-i)
    }

    return ret;
}


// # Calcolo della media da una mappa
double mean_dic(const map<int, unsigned long long int> &a) {
    double sum = 0, dot = 0;
    for (map<int, unsigned long long int>::const_iterator it = a.cbegin(); it != a.cend(); ++it) {
        sum += it->second;
        dot += it->first * it->second;
    }

    return dot / sum;
}

// # Calcolo della deviazione standard da una mappa
double std_dic(const map<int, unsigned long long int> &a) {
    double sum = 0, dot = 0, dotsq = 0;
    for (map<int, unsigned long long int>::const_iterator it = a.cbegin(); it != a.cend(); ++it) {
        sum += it->second;
        dot += it->first * it->second;
        dotsq += (it->first) * (it->first) * it->second;
    }

    dot /= sum;
    dotsq /= sum;

    return sqrt(dotsq - (dot * dot));
}

void ciclo_singola_matrice(double eps,
                           double dil,
                           double thresh,
                           mappa *isto_dist,
                           mappa *isto_lung,
                           mappa *isto_size,
                           mappa *isto_nclu,
                           mappa *isto_nvic,
                           int ripetiz,
                           bool loadNets,
                           double* nets) {
  double mat[N*N];
  for (int rip = 0; rip < ripetiz; ++rip) {
        //py::print("loadNet",loadNets);
        if(loadNets == false){
          matrice(mat,eps, dil);     // Matrice delle connessioni
        }else{
          for (unsigned int j = 0; j<N;++j){
            for (unsigned int i= 0; i<N;++i){
              mat[i+j*N] = nets[i+j*N+rip*N*N];
            }    
          }
        }

        if (show==1){
          for (unsigned int j=0; j<N;++j){
            py::list line;
            for (unsigned int i = 0; i < N; ++i) {
              line.append(mat[i+j*N]);
            }
            py::print(j,line);
          }
	      }

        int icluster = 0;                      // Contatore dei cluster
        //unordered_map<vector<int>, int, VectHash<int>> statiVisitati{};                 // Lista degli visitati
        const unsigned int numStati = (1 << N);
        map<int, int> statiVisitati;          // Lista degli visitati
        map<int, int> path;          // Lista degli visitati
        int charact[numStati];                // Carattere degli stati esplorati (1 inizio, 2 transiente, 3 ciclo limite)
        py::list allCycles;
        int whichCycle[numStati];                // C.L. degli stati
        int dist[numStati];                   // Distanza dal C.L. degli stati
        int nvic[numStati];                   // Numero stati che evolvono in un certo stato
        unsigned int si[N];                            // stati iniziale espresso come vettore binario
        //unsigned int sf[N];                            // stati finale espresso come vettore binario
        double v[N];                            // somma pesata
        unsigned int k0;
        unsigned int k;                       
        unsigned int t;
    
        for (unsigned int n = 0; n < numStati; ++n) {                      // Ciclo sui 2^N stati iniziali
           py::list cycle;
           path.clear();
           k0 = n;                           // Lista degli stati singoli di una condiz iniziale, in base 10
           if (show==1) py::print("k0",k0);
           
           if (statiVisitati.find(k0) != statiVisitati.end()) continue;  //  Controllo che uno stato come quello iniziale (ovvero [0,0...0, n]) non sia già comparso
           path[k0] = 1;
           statiVisitati[k0] = 1;      // Aggiungo alla mappa il nuovo vettore e l'indice corrispondente (ovvero la lunghezza della mappa)
           charact[k0]=1;
           nvic[k0]=0;

           t = 0;                                                     // Contatore del tempo del processo
           s_da_n(si,n);
           if (show==1){                                   // Stato come vettore di attivazione dei neuroni
            py::list vec;
            for (unsigned int i=0; i<N;++i) vec.append(si[i]);
            py::print("vec",vec);
           }


           while (true) {
               t += 1;                                                    // Lo aggiorno subito perchè nello 0 dele liste già ho messo lo stato iniziale (quindi voglio che alla prima iterazione valga 1)
               dot(mat, si, v);
               //if (show==1){                                   // Stato come vettore di attivazione dei neuroni
               //   py::list vec;
               //   for (unsigned int i=0; i<N;++i) vec.append(v[i]);
               //   py::print("weighted sum",vec);
               //}
               // Heavyside for each element in v 
               for (int i = 0; i < N; ++i) {
                   si[i] = (v[i] >= thresh);//>= thresh);  for backwords comparrison 
               }  

               k = n_da_s(si);          // Ricavo il valore decimale corrispondente allo stato
               if (show==1){  
                py::print(">k",k);                                 // Stato come vettore di attivazione dei neuroni
                py::list vec;
                for (unsigned int i=0; i<N;++i) vec.append(si[i]);
                py::print("vec",vec);
               }

               if (statiVisitati.find(k) == statiVisitati.end()) {
                  nvic[k]=1;
                  charact[k]=2;
                  statiVisitati[k]=1; //Aggiungi agli stati visitati
                  path[k]=1;
                  if (show==1){
                    py::list vec;
                    for(std::map<int,int>::iterator iter = statiVisitati.begin(); iter != statiVisitati.end(); ++iter){
                      int key =  iter->first;
                      vec.append(key);
                    }
                    py::print("statiVisitati",vec);
                  }
               } else {
                  std::map<int,int>::iterator iterPath = path.find(k);
                  if (iterPath != path.end()) {
                     icluster += 1;
                     nvic[k] += 1;
                     //all states in cycle must be assigned character 3, that says they are cycles
                     for(std::map<int,int>::iterator iterCycle = iterPath; iterCycle != path.end(); ++iterCycle){
                       int w = iterCycle->first;
                       charact[w]=3;
                       cycle.append(w);
                     } 
                     allCycles.append(cycle);
                     if (show==1){
                       py::print("cycle",cycle);
                       py::print("found cycles",allCycles);
                     }
                     //all states in path must be assigned to the corresponding cycle
                     for(std::map<int,int>::iterator iterPath2 = path.begin(); iterPath2 != path.end(); ++iterPath2){
                       whichCycle[iterPath2->first]=icluster;
                     } 
                     break;
                   }else{
                     nvic[k] += 1; 
                     int c = whichCycle[k];
                     for(std::map<int,int>::iterator iterPath2 = path.begin(); iterPath2 != path.end(); ++iterPath2){
                       if((int) k != iterPath2->first) whichCycle[iterPath2->first]=c;
                     } 
                     break;                
                   }
               }
           }

        }
        
        if (show==1){  
          py::list vec;
          for (unsigned int i=0; i<numStati;++i) vec.append(whichCycle[i]);
          py::print("vec",vec);
        }

        //mtx.lock();
        for (int i = 0; i < (1<<N); ++i) {                                  
            (*isto_nvic)[nvic[i]] += 1;                                        
//            (*isto_dist)[dist[i]] += 1;                                        
//            if (charact[i] == 1) (*isto_dmax)[dist[i]] += 1;                   
        }

        for (int i = 1; i < icluster+1; ++i) {
            int lunghezza = 0, bacino = 0;
            for (unsigned int j = 0; j < numStati; ++j) {
                lunghezza += ((whichCycle[j]==i) && (charact[j]==3));       // se lo stato j appartiene al cluster e ha carattere 3, è cl
                bacino += (whichCycle[j] == i);
            }
            (*isto_lung)[lunghezza] += 1;
            (*isto_size)[bacino] += 1;
        }

        (*isto_nclu)[icluster] += 1;
        //mtx.unlock();
  }
}


py::dict runRHNN(double eps, double dil,int ripetiz = 1000){
  mappa isto_dist;                                  // Rappresento gli istogrammi come mappe
  //mappa isto_dmax;
  mappa isto_lung;
  mappa isto_size;
  mappa isto_nclu;
  mappa isto_nvic;

  py::dict dict_measures;

  double *ptrNet;
  //ptrNet[0] = 0;
  //double eps = 0.0;
  //double dil = 0.0;
  py::print("N",N,"eps",eps, "dil",dil,"ripetiz",ripetiz);
  //int m = 1;
  //int refr = 0; 
  double thresh = 0.;
  //double tau = 1.0;
  //vector<double> f_k = generate_f_k(m, tau);        // Vettore dei decadimenti
  //int ripetiz = 1;

  ciclo_singola_matrice(eps, dil, thresh,
                        &isto_dist, &isto_lung, &isto_size, &isto_nclu, &isto_nvic,
                        ripetiz, false, ptrNet);

  // Calcola medie
  double meanDist = mean_dic(isto_dist);
  double meanSize = mean_dic(isto_size);
  double meanLung = mean_dic(isto_lung);
  double meanNClu = mean_dic(isto_nclu);
  //mettile nel dizionario
  dict_measures["Dist"] = meanDist;
  dict_measures["Size"] = meanSize;
  dict_measures["Lung"] = meanLung;
  dict_measures["NClu"] = meanNClu;

  return dict_measures;
}

py::dict runRHNNwithNets(double thresh, py::array_t<double> net){
  
  py::buffer_info bufNet = net.request();
  double *ptrNet = (double *) bufNet.ptr;
  unsigned int Np = (unsigned int) bufNet.shape[2];
  unsigned int Npp = (unsigned int)  bufNet.shape[1];
  unsigned int ripetiz = (unsigned int) bufNet.shape[0];
  py::print("N",N);
  py::print("Np",Np);
  py::print("Npp",Npp);
  assert (N==Np);  // change line 11 # define N 14
  assert (N==Npp);
  py::print("ripetiz",ripetiz);

  //for (unsigned int k=0; k<ripetiz;++k){
  //  for (unsigned int j=0; j<N;++j){
  //    py::list line;
  //    for (unsigned int i = 0; i < N; ++i) {
  //      line.append(ptrNet[i+j*N+k*N*N]);
  //      //py::print(i,j,k,ptrNet[i+j*N+k*N*N]);
  //    }
  //    py::print(j,k,line);
  //  }
	//}
  
  mappa isto_dist;                                  // Rappresento gli istogrammi come mappe
  //mappa isto_dmax;
  mappa isto_lung;
  mappa isto_size;
  mappa isto_nclu;
  mappa isto_nvic;

  py::dict dict_measures;


  //double eps = 0.0;
  //double dil = 0.0;
  py::print("N",N,"thresh",thresh,"ripetiz",ripetiz);
  //int m = 1;
  //int refr = 0; 
  double threshold = 0.;
  //double tau = 1.0;
  //vector<double> f_k = generate_f_k(m, tau);        // Vettore dei decadimenti
  //int ripetiz = 1;

  ciclo_singola_matrice(0.0, 0.0, threshold,
                        &isto_dist, &isto_lung, &isto_size, &isto_nclu, &isto_nvic,
                        ripetiz,true,ptrNet);

  // Calcola medie
  double meanDist = mean_dic(isto_dist);
  double meanSize = mean_dic(isto_size);
  double meanLung = mean_dic(isto_lung);
  double meanNClu = mean_dic(isto_nclu);
  //mettile nel dizionario
  dict_measures["Dist"] = meanDist;
  dict_measures["Size"] = meanSize;
  dict_measures["Lung"] = meanLung;
  dict_measures["NClu"] = meanNClu;

  return dict_measures;
}

PYBIND11_MODULE(hrnn, m) {
    m.doc() = "run recurrent hopfield neural network"; // optional module docstring
    m.def("runRHNN", &runRHNN, "Runs recurrent neural network");
    m.def("runRHNNwithNets", &runRHNNwithNets, "Runs recurrent neural network with a array of connettivity matrices");
}
