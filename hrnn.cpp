
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


// Definisco le funzioni dichiarate

// # Funzione per calcolare potenze di interi
/* int npow(base, esp) {
    int ret = 1;

    for (int i = 0; i < esp; i++) ret *= base;

    return ret;
} */

// # Funzione per creare la matrice
array<array<double, N>, N> matrice(double eps, double dil) { 
    array<array<double, N>, N>  mat{};
    array<array<double, N>, N>  s_mat(mat);
    array<array<double, N>, N>  a_mat(mat);
    unsigned seed = chrono::system_clock::now().time_since_epoch().count();
    default_random_engine re (seed);
    uniform_real_distribution<double> unif(0,1);
    double a1, a2; 

    for (int i = 0; i < N; i++) {
        for (int j = i + 1; j < N; j++) {
            if (unif(re) > dil){
              a1 = (unif(re) - 0.5) * 2;                         // Gli elementi sono diversi da 0 solo con prob (1-dil)
              s_mat[i][j] = a1;                                  // Creo le due matrici, una simmetrica e una asimmetrica
              s_mat[j][i] = a1;                      
            }
            if (unif(re) > dil) {
              a2 = (unif(re) - 0.5) * 2;
              a_mat[i][j] = a2;
              a_mat[j][i] = -a2;
            }
        }
    }

    for (int i = 0; i < N; i++) {
        for (int j = 0; j < N; j++){
            mat[i][j] = (1 - eps / 2) * s_mat[i][j] + (eps / 2) * a_mat[i][j];   // La diagonale viene nulla per costruzione
        }
    }

    return mat;
}

// # Calcolo il prodotto tra matrice e vettore
array<double, N> dot(const array<array<double, N>, N> &mat, const array<int, N> &s) {
    array<double, N> ret{};

    for (int i = 0; i < N; ++i) {
        for (int j = 0; j < N; ++j) {
            ret[i] += mat[i][j] * s[j];
        }
    }

    return ret;
}

// # Calcolo il prodotto tra matrice e vettore composto da 0 e 1
array<double, N> dot01(const array<array<double, N>, N> &mat, const array<int, N> &s) {
    array<double, N> ret{};

    for (int j = 0; j < N; ++j) {
        if (s[j]) {
          for (int i = 0; i < N; ++i) {
              ret[i] += mat[i][j];
          }
        }
    }

    return ret;
}

// # Calcolo il prodotto tra matrice e vettore
array<double, N> dot_vec(const vector<array<double, N>> &pot_vec, const vector<double> &f) {
    array<double, N> ret{};
    int m = f.size();

    for (int i = 0; i < m; ++i) {
        for (int j = 0; j < N; ++j) {
            ret[j] += pot_vec[i][j] * f[i];
        }
    }

    return ret;
}

// # Creo il vettore di decadimento degli stati memorizzati (quanto pesa ogni stato precedente per lo spike attuale)
vector<double> generate_f_k(int m, double tau) {         // f_k ha lunghezza che dipende da m, quindi varia: non posso usare array
    vector<double> f_k(m);

    for (int i = 0; i < m ; i++) {
        f_k[i] = exp(- tau * (m - 1 - i));                  // Gli indici più bassi hanno decadimento maggiore (e quindi un valore più basso). L'ultimo indice ha valore 1 (nessun decadimento)
    }

    return f_k;
}

// # Creo lo stato "n"
array<int, N> s_da_n(int n) {
    array<int, N> s0{};

    for (int i = 0; i < N; i++) {
        s0[N-i-1] = (int(double(n) / (1 << i))) % 2;           // 1<<i è 2**i, calcolato molto velocemente
    }

    return s0;
}

// # Calcolo "n" corrispondente allo stato "s"
int n_da_s(const array<int, N> &s) {
    int ret = 0;

    for (int i = 0; i < N; i++){
        ret += s[i] * (1 << (N - 1 - i));               // Ovvero +=s[i]*2**(N-1-i)
    }

    return ret;
}

// # Ricavo il valore hash di un vettore DA ELIMINARE, NON È PIÙ UTILE
int codif_vect(const vector<int> &vec) {
    int ret = vec.size();

    for (int i = 0; i < ret; i++) {
        ret ^= vec[i] + 0x9e3779b9 + (ret << 6) + (ret >> 2);
    }

    return ret;
}

// # Ottengo il valore hash di un mstato, ottenuto dalla lista degli stati
//vector<int> mstato_da_n(const vector<int> &s10, int m) {
//    int l = s10.size();
//
//    if (l < m){
//        vector<int> s(m-l, 0);
//        s.insert(s.end(), s10.cbegin(), s10.cend());
//        return s;
//    }
//
//    return (vector<int>(s10.cend() - m, s10.cend()));
//}

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

// # Calcolo della media da una mappa
double mean_dic(const unordered_map<int, unsigned long long int> &a) {
    double sum = 0, dot = 0;
    for (unordered_map<int, unsigned long long int>::const_iterator it = a.cbegin(); it != a.cend(); ++it) {
        sum += it->second;
        dot += it->first * it->second;
    }

    return dot / sum;
}

// # Calcolo della deviazione standard da una mappa
double std_dic(const unordered_map<int, unsigned long long int> &a) {
    double sum = 0, dot = 0, dotsq = 0;
    for (unordered_map<int, unsigned long long int>::const_iterator it = a.cbegin(); it != a.cend(); ++it) {
        sum += it->second;
        dot += it->first * it->second;
        dotsq += (it->first) * (it->first) * it->second;
    }

    dot /= sum;
    dotsq /= sum;

    return sqrt(dotsq - (dot * dot));
}



//void ciclo_singola_matrice(double eps, double dil, int m, int refr, double thresh, const vector<double> &f_k, mappa *isto_dist, mappa *isto_dmax, mappa *isto_lung, mappa *isto_size, mappa *isto_nclu, mappa *isto_nvic, int ripetiz = 1);
//mutex mtx;
//
//int main(int argc, char *argv[]){
//    if (argc < 9) {                                  // Messaggio di errore se si lancia il programma senza argomenti
//        fprintf(stderr, "SYNTAX ERROR: %s eps, dil, tau, m, refr, thresh, ripe, thr:\n"
//        "\t      - epsilon density (quante eps generare)\n"
//        "\t      - dilution density (quante dil generare)\n"
//        "\t      - tau (double; reciproco del tempo caratteristico di decadimento)\n"
//        "\t      - m (int; numero di tempi precedenti considerati)\n"
//        "\t      - refrattario (int; 1 se si vuole il periodo refrattario, 0 altrimenti)\n"
//        "\t      - threshold (double; threshold di attivazione)\n"
//        "\t      - ripetizioni (int; numero ripetizioni per matrice)\n"
//        "\t      - thr (int; numero di threads)\n", argv[0]);
//        return EXIT_FAILURE;
//    }
//    int n_eps = atof(argv[1]);                       // Dichiaro gli argomenti passati al .exe, spiegati nel messaggio di errore
//    int n_dil = atof(argv[2]);
//    double tau = atof(argv[3]);
//    int m = atoi(argv[4]);
//    int refr = atoi(argv[5]);
//    double thresh = atof(argv[6]);
//    int ripe = atoi(argv[7]);
//    int thr = atoi(argv[8]);
//    
//    double eps = 0.0;
//    double dil = 0.0;
//
//    vector<double> f_k = generate_f_k(m, tau);        // Vettore dei decadimenti
//    
//    int n_threads = thread::hardware_concurrency();
//    n_threads = n_threads < thr ? n_threads : thr;
//    int passi = int((double)(ripe + n_threads - 1) / n_threads);        // numero di batch parallelizzati 
// 
//    ofstream osd, oss, osl, osc;
//    string path = "/content/drive/My Drive/";
//    string file_D = path + "D_N=" + to_string(N) + "_Ne=" + to_string(n_eps)+ "_Nd=" + to_string(n_dil) + "_tau=" + to_string(tau) + "_m=" + to_string(m) + "_refr=" + to_string(refr) + "_thresh=" + to_string(thresh) + "_ripe=" + to_string(ripe) + ".dat" ;
//    string file_S = path + "S_N=" + to_string(N) + "_Ne=" + to_string(n_eps)+ "_Nd=" + to_string(n_dil) + "_tau=" + to_string(tau) + "_m=" + to_string(m) + "_refr=" + to_string(refr) + "_thresh=" + to_string(thresh) + "_ripe=" + to_string(ripe) + ".dat" ;
//    string file_L = path + "L_N=" + to_string(N) + "_Ne=" + to_string(n_eps)+ "_Nd=" + to_string(n_dil) + "_tau=" + to_string(tau) + "_m=" + to_string(m) + "_refr=" + to_string(refr) + "_thresh=" + to_string(thresh) + "_ripe=" + to_string(ripe) + ".dat" ;
//    string file_C = path + "C_N=" + to_string(N) + "_Ne=" + to_string(n_eps)+ "_Nd=" + to_string(n_dil) + "_tau=" + to_string(tau) + "_m=" + to_string(m) + "_refr=" + to_string(refr) + "_thresh=" + to_string(thresh) + "_ripe=" + to_string(ripe) + ".dat" ;
//    osd.open(file_D);
//    oss.open(file_S);
//    osl.open(file_L);
//    osc.open(file_C);
//    
//    for (int e = 0; e < n_eps+1; ++e) {
//        eps = (double) e / n_eps;
//        for (int d = 0; d < n_dil; ++d) {
//            dil = (double) d / n_dil;
//            
//            
//            mappa isto_dist;                                  // Rappresento gli istogrammi come mappe
//            mappa isto_dmax;
//            mappa isto_lung;
//            mappa isto_size;
//            mappa isto_nclu;
//            mappa isto_nvic;
//            
//            vector<future<void>> threads;
//            for (int i_thr = 0; i_thr < n_threads; ++i_thr) threads.emplace_back(async(std::launch::async, ciclo_singola_matrice, eps, dil, m, refr, thresh, f_k, &isto_dist, &isto_dmax, &isto_lung, &isto_size, &isto_nclu, &isto_nvic, passi));
//        
//            for (auto &a : threads) a.wait();
//            
//            osd << mean_dic(isto_dist) << "\t";
//            oss << mean_dic(isto_size) << "\t";
//            osl << mean_dic(isto_lung) << "\t";
//            osc << mean_dic(isto_nclu) << "\t";
//            osd.flush();
//            oss.flush();
//            osl.flush();
//            osc.flush();
//        }
//        
//        osd << "\n";
//        oss << "\n";
//        osl << "\n";
//        osc << "\n";
//        
//    }
//    
//    osd.close();
//    oss.close();
//    osl.close();
//    osc.close();
//    
//    return 0;
//}


void ciclo_singola_matrice(double eps,
                           double dil,
                           int m,
                           int refr, 
                           double thresh,
                           const vector<double> &f_k,
                           mappa *isto_dist,
                           mappa *isto_dmax,
                           mappa *isto_lung,
                           mappa *isto_size,
                           mappa *isto_nclu,
                           mappa *isto_nvic,
                           int ripetiz,
                           bool loadNets,
                           double* nets) {
  array<array<double, N>, N> mat;
  for (int rip = 0; rip < ripetiz; ++rip) {
        //py::print("loadNet",loadNets);
        if(loadNets == false){
            mat = matrice(eps, dil);     // Matrice delle connessioni
        }else{
            for (unsigned int j = 0; j<N;++j){
              for (unsigned int i= 0; i<N;++i){
                mat[i][j] = nets[i+j*N+rip*N*N];
              }    
            }
        }

        //for (unsigned int j=0; j<N;++j){
        //  py::list line;
        //  for (unsigned int i = 0; i < N; ++i) {
        //    line.append(mat[i][j]);
        //  }
        //  py::print(j,line);
        //}
	      

        int icluster = 0;                                       // Contatore dei cluster
        //unordered_map<vector<int>, int, VectHash<int>> statiVisitati{};                 // Lista degli visitati
        int numStati = (1 << N);
        map<int, int> statiVisitati;                 // Lista degli visitati
        int charact[numStati];                                  // Carattere degli mStati della matrice (1 inizio, 2 transiente, 3 ciclo limite)
        int cluster[numStati];                                  // Cluster degli mStati
        int dist[numStati];                                     // Distanza dal C.L. degli mStati
        int nvic[numStati];                                     // Numero di mStati che evolvono in un certo mStato
    
        for (int n = 0; n < numStati; ++n) {                      // Ciclo sui 2^N stati iniziali
           int s0 = n;                           // Lista degli stati singoli di una condiz iniziale, in base 10
           //py::print("stati10",stati10[0]);
           //vector<int> mstato_k = mstato_da_n(stati10, m);      // Primo mStato
           //py::list vec;
           //for (unsigned int i=0; i<m;++i) vec.append(mstato_k[i]);
           //py::print("vec",vec);
           
           if (statiVisitati.find(s0) != statiVisitati.end()) continue;  //  Controllo che uno stato come quello iniziale (ovvero [0,0...0, n]) non sia già comparso

           statiVisitati[s0] = n;      // Aggiungo alla mappa il nuovo vettore e l'indice corrispondente (ovvero la lunghezza della mappa)
           charact.push_back(1);
           nvic.push_back(0);

           int t = 0;                                                     // Contatore del tempo del processo
           array<int, N> s = s_da_n(n);                                   // Stato come vettore di attivazione dei neuroni
           array<int, N> s1;
           //py::list vec;
           //for (unsigned int i=0; i<N;++i) vec.append(s[i]);
           //py::print("vec",vec);
           //array<int, N> refr_s{};                                        // Array che conta quanti tempi dura ancora il refr di un neurone
           //array<double, N> attivazione;  // attivazione
           //attivazione = dot(mat, s);                             // Per lo stato iniziale assumo che gli stati precedenti fossero tutti 0. .back() mi da l'ultimo elemento
           array<double, N> v;
           unsigned int indice, prima_comparsa, k;

           while (true) {
               t += 1;                                                    // Lo aggiorno subito perchè nello 0 dele liste già ho messo lo stato iniziale (quindi voglio che alla prima iterazione valga 1)

               v = dot(mat, s);
               for (int i = 0; i < N; ++i){
                   s1[i]=0;
               }
               for (int i = 0; i < N; ++i) {
                   s1[i] = (v[i] >= thresh);                                        // Theta di Heavyside per ogni elemento di v, che ci da 
               }  

               k = n_da_s(s);                                              // Ricavo il valore decimale corrispondente allo stato
               //py::print(">k",k);
               //py::list vec;
               //for (unsigned int i=0; i<N;++i) vec.append(s[i]);
               //py::print("vec",vec);

               statiVisitati[k]=n; //Aggiungi agli stati visitati

               if (statiVisitati.find(k) != statiVisitati.end()) {
                  nvic[k]=1;
                  charact[k]=2;   
                  continue;             //  Cerco se questo stato è già comparso per questa matrice
               } else {
                  break;
               }
               
               if (it_indice == mstati_di_mat.end()) {                     // Ovvero se non è stato trovato: non è mai comparso. Lo aggiungo alle liste (vicini, carattere, mStati) 
               
               } else if (cluster.size() <= (indice = it_indice->second)) {    // Ovvero se esiste in questo percorso. Controllo se l'indice di mStati_di_mat esiste anche in cluster
                  icluster += 1;
                  nvic[indice] += 1;
                  prima_comparsa = indice - cluster.size();                      //  Dove è già comparso in questo percorso. Escludo l'ultimo stato dalla ricerca, così ottengo un solo numero
                  cluster.insert(cluster.end(), t, icluster);                    //  Gli stati di questo cluster sono gli ultimi t stati aggiunti a mstati_di_M (quelli del percorso di questa cond. iniziale), quindi i loro indici sono gli ultimi t
                  for (int i = prima_comparsa; i > 0; --i) dist.push_back(i);    // Le distanze dal ciclo (elementi fuori dal ciclo)
                  dist.insert(dist.end(), (t - prima_comparsa), 0);              // Distanze (nulle) degli elementi nel ciclo
                  fill(charact.end() - t + prima_comparsa, charact.end(), 3);    // Li caratterizzo come cicli
                  break;
               } else {                                                                         
                  nvic[indice] += 1;                                             // Ovvero se esiste in mstati_di_mat e non in questo percorso                     
                  cluster.insert(cluster.end(), t, cluster[indice]);                                  
                  for (int i = t; i > 0; --i) dist.push_back(i + dist[indice]);                          
                  break;
               }
           }
        }
        
        //mtx.lock();
        for (int i = 0; i < (1<<N); ++i) {                                  
            (*isto_nvic)[nvic[i]] += 1;                                        
            (*isto_dist)[dist[i]] += 1;                                        
            if (charact[i] == 1) (*isto_dmax)[dist[i]] += 1;                   
        }

        for (int i = 1; i < icluster+1; ++i) {
            int lunghezza = 0, bacino = 0;
            for (unsigned int j = 0; j < charact.size(); ++j) {
                lunghezza += ((cluster[j]==i) && (charact[j]==3));       // se lo stato j appartiene al cluster e ha carattere 3, è cl
                bacino += (cluster[j] == i);
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
  mappa isto_dmax;
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
  double thresh = 0.1;
  //double tau = 1.0;
  //vector<double> f_k = generate_f_k(m, tau);        // Vettore dei decadimenti
  //int ripetiz = 1;

  ciclo_singola_matrice(eps, dil, thresh,
                        &isto_dist, &isto_dmax, &isto_lung, &isto_size, &isto_nclu, &isto_nvic,
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
  mappa isto_dmax;
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
  double thresh = 0.1;
  //double tau = 1.0;
  //vector<double> f_k = generate_f_k(m, tau);        // Vettore dei decadimenti
  //int ripetiz = 1;

  ciclo_singola_matrice(0.0, 0.0, thresh,
                        &isto_dist, &isto_dmax, &isto_lung, &isto_size, &isto_nclu, &isto_nvic,
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
