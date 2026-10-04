#include "model.hpp"
#include "optim.hpp"
#include "tokenizer.hpp"
#include <chrono>
#include <fstream>
#include <iostream>
#include <random>
#include <sstream>
#include <string>
#include <algorithm>
using namespace rp;
static std::vector<int> read_all(const std::string&f,Tokenizer&tok){std::ifstream in(f);std::stringstream ss;ss<<in.rdbuf();return tok.encode(ss.str());}
static int sample(const Tensor&logits,int row,float temp,std::mt19937&rng){int V=logits->s[1];std::vector<float>p(V);float mx=-1e30f;for(int j=0;j<V;++j)mx=std::max(mx,logits->x[row*V+j]);float z=0;for(int j=0;j<V;++j){p[j]=std::exp((logits->x[row*V+j]-mx)/std::max(.05f,temp));z+=p[j];}for(float&v:p)v/=z;std::discrete_distribution<int>d(p.begin(),p.end());return d(rng);}
int main(int argc,char**argv){Config cfg;Tokenizer tok;Model model(cfg,123);if(argc<2){std::cout<<"rp_llm train <data> <steps> <checkpoint> | generate <checkpoint> <text>\n";return 0;}std::string mode=argv[1];if(mode=="train"){std::string data=argc>2?argv[2]:"data/rp.txt";int steps=argc>3?std::stoi(argv[3]):1000;std::string ck=argc>4?argv[4]:"model.bin";auto ids=read_all(data,tok);auto p=model.params();AdamW opt;opt.init(p);std::mt19937 rng(7);if(ids.size()<=cfg.ctx+1){std::cerr<<"Dataset too small\n";return 1;}auto t0=std::chrono::steady_clock::now();for(int s=1;s<=steps;++s){std::uniform_int_distribution<size_t>d(0,ids.size()-cfg.ctx-1);size_t at=d(rng);std::vector<int>x(ids.begin()+at,ids.begin()+at+cfg.ctx),y(ids.begin()+at+1,ids.begin()+at+cfg.ctx+1);auto logits=model.forward(x);float loss=loss_and_seed(logits,y);backward(logits);clip(p,1.0f);opt.update(p);if(s%10==0){std::cout<<"step "<<s<<" loss "<<loss<<"\n";if(s%100==0)model.save(ck);}}model.save(ck);auto dt=std::chrono::duration<double>(std::chrono::steady_clock::now()-t0).count();std::cout<<"saved "<<ck<<" in "<<dt<<"s\n";return 0;}if(mode=="generate"){if(argc<4)return 1;std::string ck=argv[2],prompt=argv[3];if(!model.load(ck)){std::cerr<<"checkpoint load failed\n";return 1;}auto ids=tok.encode(prompt);std::mt19937 rng(9);for(int n=0;n<120;++n){if(ids.size()>cfg.ctx)ids.erase(ids.begin());auto logits=model.forward(ids);int next=sample(logits,(int)ids.size()-1,.8f,rng);ids.push_back(next);if(next==tok.eos)break;}std::cout<<tok.decode(ids)<<"\n";return 0;}return 1;}
